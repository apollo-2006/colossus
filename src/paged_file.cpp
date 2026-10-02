#include "paged_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr char magic[8] = {'C', 'G', 'E', 'O', 'v', '0', '0', '7'};
constexpr uint32_t grid_bits = 14;
constexpr uint32_t grid_max = (1u << grid_bits) - 2;  // a step of rounding to spare

template <class T>
void write_vec(std::ofstream& f, const std::vector<T>& v) {
    const uint64_t n = v.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof n);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

template <class T>
void read_vec(std::ifstream& f, std::vector<T>& v) {
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof n);
    if (!f || n > (uint64_t(1) << 34) / sizeof(T)) throw std::runtime_error("geometry file: bad array length");
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
    if (!f) throw std::runtime_error("geometry file: ends early");
}

float sign_not_zero(float v) { return v >= 0 ? 1.0f : -1.0f; }

}  // namespace

namespace {

// octahedral: onto |x| + |y| + |z| = 1, lower half folded over, u and v from
// [-1, 1] to 11 bits.
uint32_t encode_normal(vec3 n) {
    const float l1 = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    float u = l1 > 0 ? n.x / l1 : 0, v = l1 > 0 ? n.y / l1 : 0;
    if (n.z < 0) {
        const float fu = (1 - std::abs(v)) * sign_not_zero(u), fv = (1 - std::abs(u)) * sign_not_zero(v);
        u = fu;
        v = fv;
    }
    auto bits = [](float x) { return static_cast<uint32_t>(std::lround((std::clamp(x, -1.0f, 1.0f) * 0.5f + 0.5f) * 2047)); };
    return bits(u) | (bits(v) << 11);
}

// bits enough for 0..max.
uint32_t bits_for(uint32_t max) {
    uint32_t b = 0;
    while (b < 32 && (max >> b) != 0) ++b;
    return b;
}

// appends values of a few bits each to a run of words, lowest bits first.
struct bit_writer {
    std::vector<uint32_t>& words;
    uint64_t bit = 0;
    void put(uint32_t value, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i, ++bit) {
            if (bit % 32 == 0) words.push_back(0);
            words.back() |= ((value >> i) & 1u) << (bit % 32);
        }
    }
};

// as the shaders read it: up to 32 bits at a bit offset into a run.
uint32_t read_bits(const uint32_t* run, uint64_t bit, uint32_t count) {
    if (count == 0) return 0;
    const uint64_t w = bit / 32;
    const uint32_t s = bit % 32;
    uint64_t v = run[w] >> s;
    if (s + count > 32) v |= uint64_t(run[w + 1]) << (32 - s);
    return uint32_t(v & ((uint64_t(1) << count) - 1));
}

uint32_t width(const gpu_cluster& c, int field) { return (c.level >> (8 + 4 * field)) & 15; }

}  // namespace

vec3 decode_normal(const gpu_cluster& c, const uint32_t* page, uint32_t k) {
    const uint32_t bx = width(c, 0), by = width(c, 1), bz = width(c, 2);
    const uint64_t at = uint64_t(k) * (bx + by + bz + 22) + bx + by + bz;
    const uint32_t uv = read_bits(page + c.vertex_offset, at, 22);
    const float u = float(uv & 2047) / 2047 * 2 - 1, v = float(uv >> 11) / 2047 * 2 - 1;
    vec3 n(u, v, 1 - std::abs(u) - std::abs(v));
    if (n.z < 0) {
        n.x = (1 - std::abs(v)) * sign_not_zero(u);
        n.y = (1 - std::abs(u)) * sign_not_zero(v);
    }
    return normalize(n);
}

uint32_t decode_triangle(const gpu_cluster& c, const uint32_t* page, uint32_t t) {
    const uint32_t ib = width(c, 3);
    const uint32_t v = read_bits(page + c.triangle_offset, uint64_t(t) * 3 * ib, 3 * ib);
    const uint32_t mask = (1u << ib) - 1;
    return (v & mask) | ((v >> ib) & mask) << 8 | ((v >> (2 * ib)) & mask) << 16;
}

vec3 decode_position(const paged_geometry& g, const gpu_cluster& c, const uint32_t* page, uint32_t k) {
    const uint32_t bx = width(c, 0), by = width(c, 1), bz = width(c, 2);
    const uint64_t at = uint64_t(k) * (bx + by + bz + 22);
    const uint32_t* run = page + c.vertex_offset;
    const uint32_t x = read_bits(run, at, bx), y = read_bits(run, at + bx, by), z = read_bits(run, at + bx + by, bz);
    // as the shaders compute it: integer grid point, one multiply-add.
    return {g.grid_min.x + g.grid_step * float(c.origin[0] + x), g.grid_min.y + g.grid_step * float(c.origin[1] + y),
            g.grid_min.z + g.grid_step * float(c.origin[2] + z)};
}

size_t paged_geometry::leaf_triangles() const {
    size_t n = 0;
    for (const gpu_cluster& c : clusters)
        if (cluster_level(c) == 0) n += c.triangle_count;
    return n;
}

paged_geometry page(const geometry& g) {
    paged_geometry p;
    p.bounds = g.bounds;
    p.lod_bounds = g.lod_bounds;
    p.levels = g.levels;

    // page 0 holds the roots; every group with members gets a page.
    std::unordered_map<uint32_t, uint32_t> page_of_group;
    std::vector<std::vector<uint32_t>> members(1);  // cluster numbers per page
    for (uint32_t i = 0; i < g.clusters.size(); ++i) {
        const uint32_t group = g.clusters[i].group;
        uint32_t pg = 0;
        if (group != UINT32_MAX) {
            auto [it, added] = page_of_group.emplace(group, static_cast<uint32_t>(members.size()));
            if (added) members.emplace_back();
            pg = it->second;
        }
        members[pg].push_back(i);
    }

    // file order: depth first from the roots, each page followed by the finer pages its clusters
    // stand for, so a request and the pages prefetched below it sit close together and can be
    // read in a few long runs (viewer/streamer.hpp, web/streamer.js).
    {
        std::vector<std::vector<uint32_t>> children(members.size());
        for (const gpu_cluster& c : g.clusters) {
            if (c.creator == UINT32_MAX) continue;
            const uint32_t from = c.group == UINT32_MAX ? 0 : page_of_group.at(c.group);
            const auto it = page_of_group.find(c.creator);
            if (it != page_of_group.end()) children[from].push_back(it->second);
        }
        std::vector<uint32_t> order, stack = {0};
        std::vector<uint8_t> seen(members.size(), 0);
        seen[0] = 1;
        while (!stack.empty()) {
            const uint32_t pg = stack.back();
            stack.pop_back();
            order.push_back(pg);
            auto& c = children[pg];
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
            for (auto it = c.rbegin(); it != c.rend(); ++it)  // reversed, so the first child comes out first
                if (!seen[*it]) {
                    seen[*it] = 1;
                    stack.push_back(*it);
                }
        }
        for (uint32_t pg = 0; pg < members.size(); ++pg)
            if (!seen[pg]) order.push_back(pg);  // unreachable: none expected, kept regardless
        std::vector<uint32_t> new_index(members.size());
        for (uint32_t k = 0; k < order.size(); ++k) new_index[order[k]] = k;
        for (auto& [group, pg] : page_of_group) pg = new_index[pg];
        std::vector<std::vector<uint32_t>> reordered(members.size());
        for (uint32_t pg = 0; pg < members.size(); ++pg) reordered[new_index[pg]] = std::move(members[pg]);
        members = std::move(reordered);
    }

    // grid fine enough that the largest cluster spans grid_max steps.
    vec3 lo(INFINITY, INFINITY, INFINITY);
    float largest = 0;
    for (const gpu_cluster& c : g.clusters) {
        vec3 clo(INFINITY, INFINITY, INFINITY), chi(-INFINITY, -INFINITY, -INFINITY);
        for (uint32_t k = 0; k < c.vertex_count; ++k) {
            const size_t v = g.cluster_vertices[c.vertex_offset + k];
            const vec3 q(g.positions[3 * v], g.positions[3 * v + 1], g.positions[3 * v + 2]);
            clo = min(clo, q);
            chi = max(chi, q);
        }
        lo = min(lo, clo);
        largest = std::max({largest, chi.x - clo.x, chi.y - clo.y, chi.z - clo.z});
    }
    p.grid_min = lo;
    p.grid_step = std::max(largest, 1e-12f) / grid_max;
    auto snap = [&](size_t v, int axis) {
        return static_cast<uint32_t>(std::lround((g.positions[3 * v + axis] - lo[axis]) / p.grid_step));
    };

    p.clusters = g.clusters;
    std::vector<std::vector<uint32_t>> deps(members.size());
    for (gpu_cluster& c : p.clusters) {
        c.group = c.group == UINT32_MAX ? 0 : page_of_group.at(c.group);
        auto it = c.creator == UINT32_MAX ? page_of_group.end() : page_of_group.find(c.creator);
        c.creator = it == page_of_group.end() ? no_page : it->second;
        // the finer page needs this one resident.
        if (c.creator != no_page) deps[c.creator].push_back(c.group);
    }

    // each page: its clusters' vertices and triangles, in turn.
    std::vector<uint32_t> words;
    for (uint32_t pg = 0; pg < members.size(); ++pg) {
        page_info info{};
        info.offset = uint64_t(p.data.size()) * 4;
        words.clear();
        for (uint32_t i : members[pg]) {
            const gpu_cluster& src = g.clusters[i];
            gpu_cluster& dst = p.clusters[i];
            dst.vertex_offset = static_cast<uint32_t>(words.size());
            for (int axis = 0; axis < 3; ++axis) {
                dst.origin[axis] = UINT32_MAX;
                for (uint32_t k = 0; k < src.vertex_count; ++k)
                    dst.origin[axis] = std::min(dst.origin[axis], snap(g.cluster_vertices[src.vertex_offset + k], axis));
            }
            // widths: each axis as wide as the cluster's extent, indices as the vertex count.
            uint32_t extent[3] = {0, 0, 0};
            for (uint32_t k = 0; k < src.vertex_count; ++k)
                for (int axis = 0; axis < 3; ++axis)
                    extent[axis] = std::max(extent[axis], snap(g.cluster_vertices[src.vertex_offset + k], axis) - dst.origin[axis]);
            uint32_t b[3];
            for (int axis = 0; axis < 3; ++axis) {
                b[axis] = bits_for(extent[axis]);
                if (b[axis] > grid_bits) throw std::logic_error("cluster wider than the grid allows");
            }
            const uint32_t ib = std::max(bits_for(src.vertex_count - 1), 1u);
            if (src.level > 255) throw std::logic_error("too many levels to pack");
            dst.level = src.level | b[0] << 8 | b[1] << 12 | b[2] << 16 | ib << 20;
            bit_writer vw{words};
            for (uint32_t k = 0; k < src.vertex_count; ++k) {
                const size_t v = g.cluster_vertices[src.vertex_offset + k];
                for (int axis = 0; axis < 3; ++axis) vw.put(snap(v, axis) - dst.origin[axis], b[axis]);
                vw.put(encode_normal({g.normals[3 * v], g.normals[3 * v + 1], g.normals[3 * v + 2]}), 22);
            }
            dst.triangle_offset = static_cast<uint32_t>(words.size());
            bit_writer tw{words};
            for (uint32_t t = 0; t < src.triangle_count; ++t) {
                const uint32_t tri = g.cluster_triangles[src.triangle_offset + t];
                for (int corner = 0; corner < 3; ++corner) tw.put((tri >> (8 * corner)) & 255, ib);
            }
            words.push_back(0);  // readers may touch the word after a run's last
        }
        while (words.size() % 4) words.push_back(0);
        info.size = static_cast<uint32_t>(words.size() * 4);
        info.cluster_count = static_cast<uint32_t>(members[pg].size());
        auto& d = deps[pg];
        std::sort(d.begin(), d.end());
        d.erase(std::unique(d.begin(), d.end()), d.end());
        info.dep_first = static_cast<uint32_t>(p.deps.size());
        info.dep_count = static_cast<uint32_t>(d.size());
        p.deps.insert(p.deps.end(), d.begin(), d.end());
        p.pages.push_back(info);
        p.data.insert(p.data.end(), words.begin(), words.end());
    }
    p.data_size = uint64_t(p.data.size()) * 4;
    return p;
}

void save_paged(const paged_geometry& p, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(magic, sizeof magic);
    const float b[12] = {p.bounds.center.x,     p.bounds.center.y,     p.bounds.center.z,     p.bounds.radius,
                         p.lod_bounds.center.x, p.lod_bounds.center.y, p.lod_bounds.center.z, p.lod_bounds.radius,
                         p.grid_min.x,          p.grid_min.y,          p.grid_min.z,          p.grid_step};
    f.write(reinterpret_cast<const char*>(b), sizeof b);
    write_vec(f, p.clusters);
    write_vec(f, p.pages);
    write_vec(f, p.deps);
    write_vec(f, p.levels);
    // page data last, 16-byte aligned, to map or read straight from the file.
    const uint64_t n = p.data.size() * 4;
    f.write(reinterpret_cast<const char*>(&n), sizeof n);
    while (f.tellp() % 16) f.put(0);
    f.write(reinterpret_cast<const char*>(p.data.data()), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("writing " + path + " failed");
}

paged_geometry load_paged(const std::string& path, bool with_data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char m[sizeof magic];
    f.read(m, sizeof m);
    if (!f || std::memcmp(m, magic, sizeof magic) != 0)
        throw std::runtime_error(path + " is not a geometry file (or an old one: rebuild it)");
    paged_geometry p;
    float b[12];
    f.read(reinterpret_cast<char*>(b), sizeof b);
    p.bounds = {{b[0], b[1], b[2]}, b[3]};
    p.lod_bounds = {{b[4], b[5], b[6]}, b[7]};
    p.grid_min = {b[8], b[9], b[10]};
    p.grid_step = b[11];
    read_vec(f, p.clusters);
    read_vec(f, p.pages);
    read_vec(f, p.deps);
    read_vec(f, p.levels);
    f.read(reinterpret_cast<char*>(&p.data_size), sizeof p.data_size);
    while (f.tellg() % 16) f.get();
    p.data_offset = static_cast<uint64_t>(f.tellg());
    if (!f) throw std::runtime_error(path + ": ends early");
    // every offset the shaders and streamer follow is checked here.
    for (const page_info& pg : p.pages)
        if (pg.offset % 16 || pg.offset + pg.size > p.data_size || uint64_t(pg.dep_first) + pg.dep_count > p.deps.size())
            throw std::runtime_error(path + ": page out of range");
    for (uint32_t d : p.deps)
        if (d >= p.pages.size()) throw std::runtime_error(path + ": dependency out of range");
    for (const gpu_cluster& c : p.clusters) {
        if (c.group >= p.pages.size() || (c.creator != no_page && c.creator >= p.pages.size()) ||
            c.vertex_count > cluster_max_vertices || c.triangle_count > cluster_max_triangles)
            throw std::runtime_error(path + ": cluster out of range");
        const uint32_t words = p.pages[c.group].size / 4;
        const uint32_t bx = width(c, 0), by = width(c, 1), bz = width(c, 2), ib = width(c, 3);
        if (bx > grid_bits || by > grid_bits || bz > grid_bits || ib == 0 || ib > 7 || (c.vertex_count > 0 && (c.vertex_count - 1) >> ib))
            throw std::runtime_error(path + ": cluster bit widths out of range");
        // each run, plus the word after it, which a read straddling its end touches.
        const uint64_t vertex_words = (uint64_t(c.vertex_count) * (bx + by + bz + 22) + 31) / 32 + 1;
        const uint64_t triangle_words = (uint64_t(c.triangle_count) * 3 * ib + 31) / 32 + 1;
        if (c.vertex_offset + vertex_words > words || c.triangle_offset + triangle_words > words)
            throw std::runtime_error(path + ": cluster outside its page");
    }
    if (with_data) {
        p.data.resize(p.data_size / 4);
        f.read(reinterpret_cast<char*>(p.data.data()), static_cast<std::streamsize>(p.data_size));
        if (!f) throw std::runtime_error(path + ": page data ends early");
        for (const gpu_cluster& c : p.clusters) {
            const uint32_t* base = &p.data[p.pages[c.group].offset / 4];
            for (uint32_t t = 0; t < c.triangle_count; ++t) {
                const uint32_t w = decode_triangle(c, base, t);
                if ((w & 255) >= c.vertex_count || (w >> 8 & 255) >= c.vertex_count || (w >> 16 & 255) >= c.vertex_count)
                    throw std::runtime_error(path + ": triangle index out of range");
            }
        }
    }
    return p;
}
