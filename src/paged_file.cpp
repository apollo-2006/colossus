#include "paged_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr char magic[8] = {'C', 'G', 'E', 'O', 'v', '0', '0', '6'};
constexpr uint32_t grid_bits = 14;
constexpr uint32_t grid_max = (1u << grid_bits) - 2;  // A step of rounding to spare

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

// Octahedral: onto |x| + |y| + |z| = 1, the lower half folded over, then
// each of u and v from [-1, 1] to 11 bits.
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

}  // namespace

vec3 decode_normal(const uint32_t* vertex) {
    const float u = float((vertex[1] >> 10) & 2047) / 2047 * 2 - 1, v = float(vertex[1] >> 21) / 2047 * 2 - 1;
    vec3 n(u, v, 1 - std::abs(u) - std::abs(v));
    if (n.z < 0) {
        n.x = (1 - std::abs(v)) * sign_not_zero(u);
        n.y = (1 - std::abs(u)) * sign_not_zero(v);
    }
    return normalize(n);
}

uint32_t decode_triangle(const gpu_cluster& c, const uint32_t* page, uint32_t t) {
    // As the shaders read it: the word the triangle starts in, and the next
    // if it runs over.
    const uint32_t byte = 3 * t, at = c.triangle_offset + byte / 4, shift = (byte % 4) * 8;
    uint32_t v = page[at] >> shift;
    if (shift > 8) v |= page[at + 1] << (32 - shift);
    return v & 0xffffff;
}

vec3 decode_position(const paged_geometry& g, const gpu_cluster& c, const uint32_t* vertex) {
    const uint32_t x = vertex[0] & 0x3fff, y = (vertex[0] >> 14) & 0x3fff, z = (vertex[0] >> 28) | ((vertex[1] & 0x3ff) << 4);
    // As the shaders compute it: integer grid point, then one multiply-add.
    return {g.grid_min.x + g.grid_step * float(c.origin[0] + x), g.grid_min.y + g.grid_step * float(c.origin[1] + y),
            g.grid_min.z + g.grid_step * float(c.origin[2] + z)};
}

size_t paged_geometry::leaf_triangles() const {
    size_t n = 0;
    for (const gpu_cluster& c : clusters)
        if (c.level == 0) n += c.triangle_count;
    return n;
}

paged_geometry page(const geometry& g) {
    paged_geometry p;
    p.bounds = g.bounds;
    p.lod_bounds = g.lod_bounds;
    p.levels = g.levels;

    // Page 0 is the roots; every group with members gets a page.
    std::unordered_map<uint32_t, uint32_t> page_of_group;
    std::vector<std::vector<uint32_t>> members(1);  // Cluster numbers per page
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

    // The grid: fine enough that the largest cluster spans grid_max steps.
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
        // The finer page needs this one resident.
        if (c.creator != no_page) deps[c.creator].push_back(c.group);
    }

    // Each page: its clusters' vertices and triangles, in turn.
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
            for (uint32_t k = 0; k < src.vertex_count; ++k) {
                const size_t v = g.cluster_vertices[src.vertex_offset + k];
                uint32_t d[3];
                for (int axis = 0; axis < 3; ++axis) {
                    d[axis] = snap(v, axis) - dst.origin[axis];
                    if (d[axis] > (1u << grid_bits) - 1) throw std::logic_error("cluster wider than the grid allows");
                }
                const uint32_t n = encode_normal({g.normals[3 * v], g.normals[3 * v + 1], g.normals[3 * v + 2]});
                words.push_back(d[0] | (d[1] << 14) | (d[2] << 28));
                words.push_back((d[2] >> 4) | (n << 10));
            }
            dst.triangle_offset = static_cast<uint32_t>(words.size());
            std::vector<uint8_t> bytes;
            for (uint32_t t = 0; t < src.triangle_count; ++t) {
                const uint32_t tri = g.cluster_triangles[src.triangle_offset + t];
                bytes.insert(bytes.end(), {uint8_t(tri), uint8_t(tri >> 8), uint8_t(tri >> 16)});
            }
            while (bytes.size() % 4) bytes.push_back(0);
            for (size_t b = 0; b < bytes.size(); b += 4)
                words.push_back(bytes[b] | bytes[b + 1] << 8 | bytes[b + 2] << 16 | uint32_t(bytes[b + 3]) << 24);
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
    // The page data last, 16-byte aligned, so a reader can map it or read
    // pages straight from the file.
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
    // Every offset the shaders and the streamer follow is checked here.
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
        if (uint64_t(c.vertex_offset) + 2 * c.vertex_count > words || uint64_t(c.triangle_offset) + (3 * c.triangle_count + 3) / 4 > words)
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
