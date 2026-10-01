// Tests for the builder, on procedural meshes so they need no downloads:
// PLY and OBJ reading, welding, clustering limits, the simplifier's locks
// and flips, the hierarchy's cracks at every cut, and the file round trip.
#include "cluster.hpp"
#include "dag.hpp"
#include "geometry_file.hpp"
#include "lod_check.hpp"
#include "mesh.hpp"
#include "paged_file.hpp"
#include "simplify.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>

namespace {

int failures = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            ++failures;                                                            \
        }                                                                          \
    } while (0)

// A sphere from a subdivided octahedron, its radius wobbled by a few
// sines so the simplifier has detail to remove. Wound counterclockwise
// from outside.
mesh bumpy_sphere(int subdivisions) {
    mesh m;
    m.positions = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    m.indices = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
    for (int s = 0; s < subdivisions; ++s) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> mid;
        auto midpoint = [&](uint32_t a, uint32_t b) {
            const auto key = std::minmax(a, b);
            auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            m.positions.push_back(normalize(m.positions[a] + m.positions[b]));
            return mid[key] = static_cast<uint32_t>(m.positions.size() - 1);
        };
        std::vector<uint32_t> next;
        for (size_t t = 0; t < m.indices.size(); t += 3) {
            const uint32_t a = m.indices[t], b = m.indices[t + 1], c = m.indices[t + 2];
            const uint32_t ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            next.insert(next.end(), {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca});
        }
        m.indices = std::move(next);
    }
    for (vec3& p : m.positions) {
        const float r = 1 + 0.03f * std::sin(9 * p.x) * std::sin(11 * p.y) + 0.02f * std::sin(17 * p.z + 3 * p.x);
        p = p * r;
    }
    return m;
}

// The sphere with the triangles near a few points cut away: open borders,
// as scans have.
mesh holed_sphere(int subdivisions) {
    mesh m = bumpy_sphere(subdivisions);
    const vec3 holes[] = {{0.6f, 0.6f, 0.53f}, {-0.9f, 0.1f, 0.42f}, {0, -1, 0}};
    std::vector<uint32_t> kept;
    for (size_t t = 0; t < m.indices.size(); t += 3) {
        const vec3 c = (m.positions[m.indices[t]] + m.positions[m.indices[t + 1]] + m.positions[m.indices[t + 2]]) * (1.0f / 3);
        bool cut = false;
        for (const vec3& h : holes) cut |= length(c - h) < 0.2f;
        if (!cut) kept.insert(kept.end(), &m.indices[t], &m.indices[t + 3]);
    }
    m.indices = std::move(kept);
    weld(m);  // Drops the vertices the holes left unused
    return m;
}

// A flat n by n grid of squares, two triangles each: all border.
mesh grid(int n) {
    mesh m;
    for (int y = 0; y <= n; ++y)
        for (int x = 0; x <= n; ++x) m.positions.push_back({float(x) / n, 0, float(y) / n});
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            const uint32_t a = y * (n + 1) + x, b = a + 1, c = a + n + 1, d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
        }
    return m;
}

std::string temp_path(const char* name) {
    return std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/colossus_" + name;
}

void test_ply_and_obj() {
    std::printf("PLY and OBJ reading\n");
    // One square as two triangles, in each format; the OBJ as a quad.
    const float v[4][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    const std::string head = "ply\nformat %s 1.0\ncomment test\nelement vertex 4\nproperty float x\nproperty float y\n"
                             "property float z\nproperty uchar intensity\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n";
    for (const char* format : {"ascii", "binary_little_endian", "binary_big_endian"}) {
        char header[512];
        std::snprintf(header, sizeof header, head.c_str(), format);
        std::string body;
        const int faces[2][3] = {{0, 1, 2}, {0, 2, 3}};
        auto put = [&](const void* p, size_t n) {
            std::string b(static_cast<const char*>(p), n);
            if (!std::strcmp(format, "binary_big_endian")) std::reverse(b.begin(), b.end());
            body += b;
        };
        for (auto& p : v) {
            if (!std::strcmp(format, "ascii")) body += std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + " 7\n";
            else {
                for (float c : p) put(&c, 4);
                body += char(7);
            }
        }
        for (auto& f : faces) {
            if (!std::strcmp(format, "ascii")) body += "3 " + std::to_string(f[0]) + " " + std::to_string(f[1]) + " " + std::to_string(f[2]) + "\n";
            else {
                body += char(3);
                for (int i : f) put(&i, 4);
            }
        }
        const std::string path = temp_path("square.ply");
        std::ofstream(path, std::ios::binary) << header << body;
        const mesh m = load_mesh(path);
        CHECK(m.positions.size() == 4 && m.triangle_count() == 2);
        CHECK(m.positions.size() == 4 && m.positions[2].x == 1 && m.positions[2].y == 1);
        CHECK(m.indices == std::vector<uint32_t>({0, 1, 2, 0, 2, 3}));
    }
    const std::string path = temp_path("square.obj");
    std::ofstream(path) << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1 -1/1\n";
    const mesh m = load_mesh(path);
    CHECK(m.indices == std::vector<uint32_t>({0, 1, 2, 0, 2, 3}));
}

void test_weld() {
    std::printf("welding\n");
    mesh m;  // Two triangles with their own copies of the shared edge, one repeated, one degenerate
    m.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    m.indices = {0, 1, 2, 3, 4, 5, 1, 2, 0, 0, 0, 1};
    weld(m);
    CHECK(m.positions.size() == 4);
    CHECK(m.triangle_count() == 2);
}

void test_clusters() {
    std::printf("clustering\n");
    const mesh m = bumpy_sphere(6);  // 32k triangles
    const auto clusters = clusterize(m.positions, m.indices);
    std::vector<int> seen(m.triangle_count(), 0);
    size_t over = 0;
    for (const auto& c : clusters) {
        std::set<uint32_t> verts;
        for (uint32_t t : c) {
            ++seen[t];
            verts.insert(&m.indices[3 * t], &m.indices[3 * t + 3]);
        }
        over += c.size() > cluster_max_triangles || verts.size() > cluster_max_vertices;
    }
    CHECK(over == 0);
    CHECK(std::all_of(seen.begin(), seen.end(), [](int k) { return k == 1; }));
    const double fill = double(m.triangle_count()) / clusters.size();
    std::printf("  %zu clusters, %.1f triangles each\n", clusters.size(), fill);
    CHECK(fill > 100);
}

void test_simplify() {
    std::printf("simplification\n");
    const mesh m = grid(40);  // 3200 triangles, flat, with a border
    std::vector<uint8_t> locked(m.positions.size(), 0);
    // Lock one row of vertices across the middle.
    for (int x = 0; x <= 40; ++x) locked[20 * 41 + x] = 1;
    const simplify_result r = simplify(m.positions, m.indices, locked, 400);
    std::printf("  3200 -> %zu triangles, error %.3g\n", r.indices.size() / 3, r.error);
    CHECK(r.indices.size() / 3 <= 400);
    // Flat, so nothing should be lost: the error is near zero, every
    // triangle still faces up, and the area is unchanged.
    CHECK(r.error < 1e-4f);
    double area = 0;
    bool flipped = false;
    for (size_t t = 0; t < r.indices.size(); t += 3) {
        const vec3 n = cross(m.positions[r.indices[t + 1]] - m.positions[r.indices[t]], m.positions[r.indices[t + 2]] - m.positions[r.indices[t]]);
        area += 0.5 * length(n);
        flipped |= n.y <= 0;  // grid() winds counterclockwise from above
    }
    CHECK(!flipped);
    CHECK(std::abs(area - 1.0) < 1e-4);
    // Every locked vertex is still used, still where it was.
    std::set<uint32_t> used(r.indices.begin(), r.indices.end());
    for (int x = 0; x <= 40; ++x) CHECK(used.count(20 * 41 + x) == 1);
    // The four corners cannot move: they are the only way to keep the outline.
    for (uint32_t corner : {0u, 40u, 40u * 41, 41u * 41 - 1}) CHECK(used.count(corner) == 1);
}

void test_hierarchy(const char* name, const mesh& input) {
    std::printf("hierarchy: %s, %zu triangles\n", name, input.triangle_count());
    mesh m = input;
    normalize_placement(m, false);
    const lod_mesh lod = build_lod(m, false);
    const geometry g = pack(lod);
    CHECK(g.leaf_triangles() == m.triangle_count());
    size_t roots = 0;
    for (const gpu_cluster& c : g.clusters) {
        roots += c.parent_error >= 1e30f;
        CHECK(c.lod_error <= c.parent_error);
    }
    std::printf("  %zu clusters over %zu levels, %zu roots\n", g.clusters.size(), g.levels.size(), roots);
    CHECK(g.levels.size() >= 5);
    CHECK(roots < 8);
    size_t cracked = 0;
    double worst_area = 1;
    for (const cut_report& r : check_cuts(g, 30)) {
        cracked += r.cracked_edges;
        if (r.triangles > 1000) worst_area = std::min(worst_area, r.area_ratio);
    }
    std::printf("  cracked edges over 31 cuts: %zu; smallest area ratio above 1000 triangles: %.4f\n", cracked, worst_area);
    CHECK(cracked == 0);
    CHECK(worst_area > 0.97);

    // Paged and written, then read back: every cluster's vertices and
    // triangles come out of its page as they went in, and every page's
    // dependencies are coarser pages.
    const paged_geometry paged = page(g);
    const std::string path = temp_path("test.cgeo");
    save_paged(paged, path);
    const paged_geometry back = load_paged(path, true);
    CHECK(back.clusters.size() == g.clusters.size() && back.pages.size() == paged.pages.size());
    size_t wrong = 0;
    float worst_normal = 1;
    for (size_t i = 0; i < g.clusters.size(); ++i) {
        const gpu_cluster& c = g.clusters[i];
        const gpu_cluster& pc = back.clusters[i];
        const uint32_t* base = &back.data[back.pages[pc.group].offset / 4];
        for (uint32_t k = 0; k < c.vertex_count; ++k) {
            const uint32_t v = g.cluster_vertices[c.vertex_offset + k];
            float xyz[3];
            std::memcpy(xyz, base + pc.vertex_offset + 4 * k, 12);
            wrong += xyz[0] != g.positions[3 * v] || xyz[1] != g.positions[3 * v + 1] || xyz[2] != g.positions[3 * v + 2];
            const vec3 n = decode_normal(base[pc.vertex_offset + 4 * k + 3]);
            worst_normal = std::min(worst_normal, dot(n, vec3(g.normals[3 * v], g.normals[3 * v + 1], g.normals[3 * v + 2])));
        }
        for (uint32_t t = 0; t < c.triangle_count; ++t) wrong += base[pc.triangle_offset + t] != g.cluster_triangles[c.triangle_offset + t];
    }
    std::printf("  %zu pages; normals within %.4f degrees\n", back.pages.size(), std::acos(std::min(1.0f, worst_normal)) * 57.2958f);
    CHECK(wrong == 0);
    CHECK(worst_normal > 0.99999f);
    size_t bad_deps = 0;
    for (const gpu_cluster& c : back.clusters)
        if (c.creator != no_page) {
            const page_info& fine = back.pages[c.creator];
            bad_deps += std::find(&back.deps[fine.dep_first], &back.deps[fine.dep_first] + fine.dep_count, c.group) ==
                        &back.deps[fine.dep_first] + fine.dep_count;
        }
    CHECK(bad_deps == 0);
    CHECK(back.pages[0].dep_count == 0);
}

}  // namespace

int main() {
    test_ply_and_obj();
    test_weld();
    test_clusters();
    test_simplify();
    test_hierarchy("closed sphere", bumpy_sphere(7));
    test_hierarchy("sphere with holes", holed_sphere(7));
    test_hierarchy("flat grid", grid(250));
    if (failures) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
}
