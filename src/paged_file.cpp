#include "paged_file.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr char magic[8] = {'C', 'G', 'E', 'O', 'v', '0', '0', '9'};
constexpr uint32_t uv_max = 65535;
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

uint32_t width(const gpu_cluster& c, int field) { return (c.level >> (8 + 4 * field)) & (field == 3 ? 7 : 15); }

// words from a cluster's vertex run to its texture coordinates.
uint32_t uv_run(const gpu_cluster& c) {
    return (c.vertex_count * (width(c, 0) + width(c, 1) + width(c, 2) + 22) + 31) / 32 + 1;
}

// words from a cluster's vertex run to its skin: past the texture coordinates if any.
uint32_t skin_run(const gpu_cluster& c, const uint32_t* page) {
    const uint32_t at = uv_run(c);
    if (!cluster_textured(c)) return at;
    const uint32_t widths = page[c.vertex_offset + at + 1];
    const uint32_t bu = std::min(widths & 31, 16u), bv = std::min(widths >> 5 & 31, 16u), bm = std::min(widths >> 10 & 7, 7u);
    return at + 2 + (c.vertex_count * (bu + bv + bm) + 31) / 32;
}

}  // namespace

namespace {

vec3 decode_octahedral(uint32_t uv) {
    const float u = float(uv & 2047) / 2047 * 2 - 1, v = float((uv >> 11) & 2047) / 2047 * 2 - 1;
    vec3 n(u, v, 1 - std::abs(u) - std::abs(v));
    if (n.z < 0) {
        n.x = (1 - std::abs(v)) * sign_not_zero(u);
        n.y = (1 - std::abs(u)) * sign_not_zero(v);
    }
    return normalize(n);
}

}  // namespace

vec3 decode_normal(const gpu_cluster& c, const uint32_t* page, uint32_t k) {
    const uint32_t bx = width(c, 0), by = width(c, 1), bz = width(c, 2);
    const uint64_t at = uint64_t(k) * (bx + by + bz + 22) + bx + by + bz;
    return decode_octahedral(read_bits(page + c.vertex_offset, at, 22));
}

vec2 decode_uv(const gpu_cluster& c, const uint32_t* page, uint32_t k, uint32_t* material) {
    const uint32_t* run = page + c.vertex_offset + uv_run(c);
    const uint32_t bu = run[1] & 31, bv = run[1] >> 5 & 31, bm = run[1] >> 10 & 7;
    const uint64_t at = 64 + uint64_t(k) * (bu + bv + bm);
    if (material) *material = (run[1] >> 16) + read_bits(run, at + bu + bv, bm);
    return {float((run[0] & 0xffff) + read_bits(run, at, bu)) / uv_max, float((run[0] >> 16) + read_bits(run, at + bu, bv)) / uv_max};
}

void decode_skin(const gpu_cluster& c, const uint32_t* page, uint32_t k, uint32_t& joints, uint32_t& weights) {
    const uint32_t* run = page + c.vertex_offset + skin_run(c, page);
    joints = run[2 * k];
    weights = run[2 * k + 1];
}

packed_cluster pack_cluster(const gpu_cluster& c) {
    packed_cluster p{};
    std::memcpy(p.center, c.center, sizeof p.center);
    p.radius = c.radius;
    p.cone = 1023u << 22;  // no cone
    if (c.cone_cutoff < 1) {
        const vec3 axis = normalize(vec3(c.cone_axis[0], c.cone_axis[1], c.cone_axis[2]));
        const uint32_t uv = encode_normal(axis);
        // raised by the axis's rounding (a test against the rounded axis is off by at most
        // that), then rounded up.
        const float cutoff = c.cone_cutoff + length(axis - decode_octahedral(uv)) + 1e-6f;
        const float steps = std::ceil((cutoff + 1) * 0.5f * 1023);
        if (steps < 1023) p.cone = uv | static_cast<uint32_t>(std::max(steps, 0.0f)) << 22;
    }
    if (c.vertex_offset > 0xffff || c.triangle_offset > 0xffff) throw std::logic_error("page too long to pack offsets");
    p.offsets = c.vertex_offset | c.triangle_offset << 16;
    if (c.level >> 24 || c.vertex_count > 255 || c.triangle_count > 255) throw std::logic_error("cluster too large to pack");
    p.level = c.level | c.vertex_count << 24;
    p.group = c.group;
    p.creator = c.creator;
    for (int axis = 0; axis < 3; ++axis) {
        if (c.origin[axis] >> 24) throw std::logic_error("grid too fine to pack");
        p.origin[axis] = c.origin[axis];
    }
    p.origin[0] |= c.triangle_count << 24;
    return p;
}

gpu_cluster unpack_cluster(const packed_cluster& p, const std::vector<page_bounds>& shared_bounds) {
    gpu_cluster c{};
    std::memcpy(c.center, p.center, sizeof c.center);
    c.radius = p.radius;
    const uint32_t steps = p.cone >> 22;
    const vec3 axis = steps == 1023 ? vec3(0, 0, 1) : decode_octahedral(p.cone);
    c.cone_axis[0] = axis.x;
    c.cone_axis[1] = axis.y;
    c.cone_axis[2] = axis.z;
    c.cone_cutoff = float(steps) / 1023 * 2 - 1;
    c.vertex_offset = p.offsets & 0xffff;
    c.triangle_offset = p.offsets >> 16;
    c.level = p.level & 0xffffff;
    c.vertex_count = p.level >> 24;
    c.group = p.group;
    c.creator = p.creator;
    for (int axis = 0; axis < 3; ++axis) c.origin[axis] = p.origin[axis] & 0xffffff;
    c.triangle_count = p.origin[0] >> 24;
    const page_bounds& parent = shared_bounds.at(p.group);
    std::memcpy(c.parent_center, parent.center, sizeof c.parent_center);
    c.parent_radius = parent.radius;
    c.parent_error = parent.error;
    if (p.creator == no_page) {
        std::memcpy(c.lod_center, p.center, sizeof c.lod_center);
        c.lod_radius = p.radius;
        c.lod_error = 0;
    } else {
        const page_bounds& lod = shared_bounds.at(p.creator);
        std::memcpy(c.lod_center, lod.center, sizeof c.lod_center);
        c.lod_radius = lod.radius;
        c.lod_error = lod.error;
    }
    return c;
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

    // grid fine enough that the largest cluster spans grid_max steps.
    vec3 lo(INFINITY, INFINITY, INFINITY);
    float largest = 0;
    for (const gpu_cluster& c : g.clusters) {
        vec3 clo(INFINITY, INFINITY, INFINITY), chi(-INFINITY, -INFINITY, -INFINITY);
        for (uint32_t k = 0; k < c.vertex_count; ++k) {
            const size_t v = g.vertex_of(g.cluster_vertices[c.vertex_offset + k]);
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
                    dst.origin[axis] = std::min(dst.origin[axis], snap(g.vertex_of(g.cluster_vertices[src.vertex_offset + k]), axis));
            }
            // widths: each axis as wide as the cluster's extent, indices as the vertex count.
            uint32_t extent[3] = {0, 0, 0};
            for (uint32_t k = 0; k < src.vertex_count; ++k)
                for (int axis = 0; axis < 3; ++axis)
                    extent[axis] = std::max(extent[axis], snap(g.vertex_of(g.cluster_vertices[src.vertex_offset + k]), axis) - dst.origin[axis]);
            uint32_t b[3];
            for (int axis = 0; axis < 3; ++axis) {
                b[axis] = bits_for(extent[axis]);
                if (b[axis] > grid_bits) throw std::logic_error("cluster wider than the grid allows");
            }
            const uint32_t ib = std::max(bits_for(src.vertex_count - 1), 1u);
            if (src.level > 255) throw std::logic_error("too many levels to pack");
            if (ib > 7) throw std::logic_error("cluster too large to pack");
            dst.level = src.level | b[0] << 8 | b[1] << 12 | b[2] << 16 | ib << 20 | uint32_t(g.textured()) << 23;
            bit_writer vw{words};
            for (uint32_t k = 0; k < src.vertex_count; ++k) {
                const size_t v = g.vertex_of(g.cluster_vertices[src.vertex_offset + k]);
                for (int axis = 0; axis < 3; ++axis) vw.put(snap(v, axis) - dst.origin[axis], b[axis]);
                vw.put(encode_normal({g.normals[3 * v], g.normals[3 * v + 1], g.normals[3 * v + 2]}), 22);
            }
            if (g.textured()) {
                words.push_back(0);  // the word after the vertex run
                // corner and widths, then each vertex's offsets.
                // coordinates as fractions of the model's range (geometry::uv_min, uv_extent),
                // and materials: the cluster's lowest and each vertex's offset from it.
                uint32_t q[3][cluster_max_vertices], lo[3] = {uv_max, uv_max, 255}, hi[3] = {0, 0, 0};
                for (uint32_t k = 0; k < src.vertex_count; ++k) {
                    const uint32_t w = g.cluster_vertices[src.vertex_offset + k];
                    const vec2 uv = g.wedge_uvs[w];
                    q[0][k] = uint32_t(std::lround(std::clamp((uv.x - g.uv_min.x) / g.uv_extent, 0.0f, 1.0f) * uv_max));
                    q[1][k] = uint32_t(std::lround(std::clamp((uv.y - g.uv_min.y) / g.uv_extent, 0.0f, 1.0f) * uv_max));
                    q[2][k] = g.wedge_material[w];
                    for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], q[a][k]); hi[a] = std::max(hi[a], q[a][k]); }
                }
                const uint32_t bu = bits_for(hi[0] - lo[0]), bv = bits_for(hi[1] - lo[1]), bm = bits_for(hi[2] - lo[2]);
                if (bm > 7) throw std::logic_error("a cluster's materials span too far to pack");
                words.push_back(lo[0] | lo[1] << 16);
                words.push_back(bu | bv << 5 | bm << 10 | lo[2] << 16);
                bit_writer uw{words};
                for (uint32_t k = 0; k < src.vertex_count; ++k) {
                    uw.put(q[0][k] - lo[0], bu);
                    uw.put(q[1][k] - lo[1], bv);
                    uw.put(q[2][k] - lo[2], bm);
                }
            }
            if (g.skinned()) {
                if (!g.textured()) words.push_back(0);  // the word after the vertex run
                for (uint32_t k = 0; k < src.vertex_count; ++k) {
                    const size_t v = 4 * size_t(g.vertex_of(g.cluster_vertices[src.vertex_offset + k]));
                    uint32_t j = 0, w = 0;
                    for (int i = 0; i < 4; ++i) {
                        j |= uint32_t(g.skin_joints[v + i]) << (8 * i);
                        w |= uint32_t(g.skin_weights[v + i]) << (8 * i);
                    }
                    words.push_back(j);
                    words.push_back(w);
                }
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

    // what each page's clusters share: their parent bounds and error, the same as the lod
    // bounds and error of the clusters made from them (src/dag.cpp). checked, so packing
    // loses none of it.
    p.shared_bounds.assign(p.pages.size(), page_bounds{{0, 0, 0}, -1, -1});
    p.shared_bounds[0] = {{p.bounds.center.x, p.bounds.center.y, p.bounds.center.z}, p.bounds.radius, FLT_MAX};
    auto share = [&](uint32_t page, const float* center, float radius, float error) {
        page_bounds& b = p.shared_bounds[page];
        if (b.radius < 0) b = {{center[0], center[1], center[2]}, radius, error};
        else if (std::memcmp(b.center, center, sizeof b.center) != 0 || b.radius != radius || b.error != error)
            throw std::logic_error("a group's clusters disagree on their bounds");
    };
    for (const gpu_cluster& c : p.clusters) {
        if (c.group != 0) share(c.group, c.parent_center, c.parent_radius, c.parent_error);
        if (c.creator != no_page) share(c.creator, c.lod_center, c.lod_radius, c.lod_error);
    }
    for (const page_bounds& b : p.shared_bounds)
        if (b.radius < 0) throw std::logic_error("a page without bounds");
    // what is drawn is snapped to the grid: each vertex moves at most half a step's
    // diagonal, so the surface does too. the errors bound the drawn surface's distance
    // from the original, so they carry it.
    const float snapping = p.grid_step * std::sqrt(3.0f) * 0.5f;
    for (size_t pg = 1; pg < p.shared_bounds.size(); ++pg) p.shared_bounds[pg].error += snapping;
    // skinned: per page the joints its group's vertices use, which bound how far skinning
    // moves its spheres. a cluster's lod sphere is its creator page's and its parent sphere its
    // own page's, so both must only grow toward the roots, as the spheres do: a page takes in
    // the pages its clusters were made from, until nothing changes.
    if (g.skinned()) {
        p.skin.assign(p.pages.size(), page_skin{});
        for (uint32_t pg = 0; pg < members.size(); ++pg) {
            uint64_t mask = 0;
            for (uint32_t i : members[pg]) {
                const gpu_cluster& c = g.clusters[i];
                for (uint32_t k = 0; k < c.vertex_count; ++k) {
                    const size_t v = 4 * size_t(g.vertex_of(g.cluster_vertices[c.vertex_offset + k]));
                    for (int a = 0; a < 4; ++a)
                        if (g.skin_weights[v + a]) {
                            if (g.skin_joints[v + a] >= 64) throw std::logic_error("joint past 64");
                            mask |= uint64_t(1) << g.skin_joints[v + a];
                        }
                }
            }
            p.skin[pg] = {mask};
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (const gpu_cluster& c : p.clusters) {
                if (c.creator == no_page) continue;
                page_skin& to = p.skin[c.group];
                const page_skin& from = p.skin[c.creator];
                if ((to.joints | from.joints) != to.joints) {
                    to.joints |= from.joints;
                    changed = true;
                }
            }
        }
    }
    p.packed.reserve(p.clusters.size());
    for (gpu_cluster& c : p.clusters) {
        p.packed.push_back(pack_cluster(c));
        c = unpack_cluster(p.packed.back(), p.shared_bounds);
    }
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
    write_vec(f, p.packed);
    write_vec(f, p.shared_bounds);
    write_vec(f, p.pages);
    write_vec(f, p.deps);
    write_vec(f, p.levels);
    // page data last, 16-byte aligned, to map or read straight from the file.
    const uint64_t n = p.data.size() * 4;
    f.write(reinterpret_cast<const char*>(&n), sizeof n);
    while (f && f.tellp() % 16) f.put(0);  // a failed stream's tellp is -1
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
    read_vec(f, p.packed);
    read_vec(f, p.shared_bounds);
    read_vec(f, p.pages);
    read_vec(f, p.deps);
    read_vec(f, p.levels);
    if (p.packed.empty()) throw std::runtime_error(path + ": no clusters");
    if (p.shared_bounds.size() != p.pages.size()) throw std::runtime_error(path + ": page bounds out of range");
    for (const packed_cluster& c : p.packed)
        if (c.group >= p.pages.size() || (c.creator != no_page && c.creator >= p.pages.size()))
            throw std::runtime_error(path + ": cluster out of range");
    p.clusters.reserve(p.packed.size());
    for (const packed_cluster& c : p.packed) p.clusters.push_back(unpack_cluster(c, p.shared_bounds));
    f.read(reinterpret_cast<char*>(&p.data_size), sizeof p.data_size);
    while (f && f.tellg() % 16) f.get();  // past the end, tellg is -1
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
        uint64_t vertex_words = (uint64_t(c.vertex_count) * (bx + by + bz + 22) + 31) / 32 + 1;
        // a textured cluster's corner and width words; the offsets are checked once read.
        if (cluster_textured(c)) vertex_words += 2;
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
            if (cluster_textured(c)) {
                const uint32_t widths = base[c.vertex_offset + uv_run(c) + 1];
                const uint32_t bu = widths & 31, bv = widths >> 5 & 31, bm = widths >> 10 & 7;
                const uint64_t end = c.vertex_offset + uv_run(c) + 2 + (uint64_t(c.vertex_count) * (bu + bv + bm) + 31) / 32 + 1;
                if (bu > 16 || bv > 16 || (widths >> 13 & 7) || end > p.pages[c.group].size / 4)
                    throw std::runtime_error(path + ": texture coordinates out of range");
            }
        }
    }
    return p;
}
