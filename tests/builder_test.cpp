// builder tests on procedural meshes (no downloads): ply and obj reading,
// welding, cluster limits, simplifier locks and flips, cracks at every cut,
// file round trip.
#include "cluster.hpp"
#include "dag.hpp"
#include "deviation.hpp"
#include "geometry_file.hpp"
#include "lod_check.hpp"
#include "mesh.hpp"
#include "paged_file.hpp"
#include "simplify.hpp"
#include "texture_file.hpp"
#include "gltf.hpp"
#include "skeleton.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <random>
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

// sphere from a subdivided octahedron, radius wobbled by a few sines for
// detail. counterclockwise from outside.
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

// the sphere with triangles near a few points cut away: borders, like scans.
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
    weld(m);  // drops vertices the holes left unused
    return m;
}

// flat n x n grid of squares, two triangles each: all border.
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

// textured: a chart per cell of a grid of side `cells` (by triangle centroid; side 2 gives the
// octants). a vertex's coordinates in chart c are ((c + 0.45 x + 0.5) / charts, 0.45 z + 0.5):
// the chart reads back from u, so a smeared seam shows as a triangle of mixed charts.
uint32_t texture_charts = 8;
vec2 chart_uv(vec3 p, uint32_t chart) { return {(chart + 0.45f * p.x + 0.5f) / texture_charts, 0.45f * p.z + 0.5f}; }
uint32_t chart_from_u(float u) { return uint32_t(u * texture_charts); }
mesh grid_textured(mesh m, uint32_t cells) {
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> wedge_of;
    for (size_t t = 0; t < m.triangle_count(); ++t) {
        const vec3 c = (m.positions[m.indices[3 * t]] + m.positions[m.indices[3 * t + 1]] + m.positions[m.indices[3 * t + 2]]) * (1.0f / 3);
        auto cell = [&](float x) { return std::min(uint32_t(std::max(0.0f, (x + 1.1f) / 2.2f * cells)), cells - 1); };
        const uint32_t chart = cell(c.x) + cells * (cell(c.y) + cells * cell(c.z));
        for (int k = 0; k < 3; ++k) {
            const uint32_t v = m.indices[3 * t + k];
            auto [it, added] = wedge_of.emplace(std::make_pair(v, chart), uint32_t(m.wedge_uvs.size()));
            if (added) {
                m.wedge_uvs.push_back(chart_uv(m.positions[v], chart));
                m.wedge_vertex.push_back(v);
                m.wedge_material.push_back(0);
            }
            m.corners.push_back(it->second);
        }
    }
    m.materials.push_back(material{});
    return m;
}

// foliage: islands no edge collapse may remove, scattered through a ball: small closed
// tetrahedra and open cards (two triangles, a hole's rim all round), like needles and leaves.
mesh islands(int count, unsigned seed, bool cards_only = false) {
    mesh m;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1, 1);
    for (int k = 0; k < count; ++k) {
        vec3 c;
        do c = vec3(u(rng), u(rng), u(rng));
        while (length(c) > 1);
        const vec3 a(u(rng), u(rng), u(rng)), b(u(rng), u(rng), u(rng));
        const float r = 0.02f;
        const uint32_t base = uint32_t(m.positions.size());
        if (k % 2 && !cards_only) {  // a tetrahedron
            m.positions.insert(m.positions.end(), {c + a * r, c + b * r, c + cross(a, b) * r, c - (a + b) * (r * 0.5f)});
            m.indices.insert(m.indices.end(), {base, base + 1, base + 2, base, base + 3, base + 1, base + 1, base + 3, base + 2,
                                               base + 2, base + 3, base});
        } else {  // a card
            m.positions.insert(m.positions.end(), {c - a * r, c + a * r, c + a * r + b * r * 3.0f, c - a * r + b * r * 3.0f});
            m.indices.insert(m.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }
    }
    return m;
}

std::string temp_path(const char* name) {
    return std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/colossus_" + name;
}

void test_ply_and_obj() {
    std::printf("PLY and OBJ reading\n");
    // one square as two triangles in each format; the obj as a quad.
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
    CHECK(m.textured() && m.wedge_uvs.size() == 4);  // one vt at four positions splits

    // texture coordinates: vertex 1 on a seam (two vts), vertex 3's shared; v flipped; the
    // texture from the material library, beside the obj.
    std::ofstream(temp_path("square.mtl")) << "newmtl m\nmap_Kd tex.jpg \n";
    std::ofstream(path) << "mtllib colossus_square.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                           "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0.25\nf 1/1 2/2 3/3\nf 1/5/1 3/3/1 4/4/1\n";
    mesh t = load_mesh(path);
    weld(t);
    CHECK(t.textured() && t.wedge_uvs.size() == 5 && t.corners.size() == 6);
    if (t.textured() && t.corners.size() == 6) {
        const vec2 a = t.wedge_uvs[t.corners[0]], b = t.wedge_uvs[t.corners[3]], c = t.wedge_uvs[t.corners[2]];
        CHECK(a.x == 0 && a.y == 1);
        CHECK(b.x == 0.5f && b.y == 0.75f);
        CHECK(t.corners[2] == t.corners[4] && c.x == 1 && c.y == 0);
        CHECK(t.wedge_vertex[t.corners[0]] == t.indices[0] && t.wedge_vertex[t.corners[3]] == t.indices[3]);
    }
    CHECK(t.materials.size() == 1 && t.materials[0].texture == temp_path("").substr(0, temp_path("").find_last_of('/') + 1) + "tex.jpg");
}

void test_weld() {
    std::printf("welding\n");
    mesh m;  // two triangles with their own copies of the shared edge, one repeated, one degenerate
    m.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    m.indices = {0, 1, 2, 3, 4, 5, 1, 2, 0, 0, 0, 1};
    weld(m);
    CHECK(m.positions.size() == 4);
    CHECK(m.triangle_count() == 2);
}

// an empty model trims to itself, with no errors to search.
void test_trim_empty() {
    std::printf("trim of an empty model\n");
    const geometry g = trim(geometry{}, 10);
    CHECK(g.clusters.empty());
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
    const mesh m = grid(40);  // 3200 triangles, flat, bordered
    std::vector<uint8_t> locked(m.positions.size(), 0);
    // lock a row of vertices across the middle.
    for (int x = 0; x <= 40; ++x) locked[20 * 41 + x] = 1;
    const simplify_result r = simplify(m.positions, m.indices, locked, 400);
    std::printf("  3200 -> %zu triangles, error %.3g\n", r.indices.size() / 3, r.error);
    CHECK(r.indices.size() / 3 <= 400);
    // flat, so nothing lost: error near zero, all triangles face up, area
    // unchanged.
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
    // every locked vertex still used, still in place.
    std::set<uint32_t> used(r.indices.begin(), r.indices.end());
    for (int x = 0; x <= 40; ++x) CHECK(used.count(20 * 41 + x) == 1);
    // the four corners hold the outline: they cannot move.
    for (uint32_t corner : {0u, 40u, 40u * 41, 41u * 41 - 1}) CHECK(used.count(corner) == 1);
}

// exact: texture coordinates checked exact (no seam had to move).
// min_area: the smallest share of the surface a cut above 1000 triangles keeps (foliage thins:
// merged islands cover less).
void test_hierarchy(const char* name, const mesh& input, bool exact = true, double min_area = 0.97) {
    std::printf("hierarchy: %s, %zu triangles\n", name, input.triangle_count());
    mesh m = input;
    // texture coordinates come from the unplaced positions: keep them.
    const std::vector<vec3> unplaced = m.positions;
    normalize_placement(m, false);
    const lod_mesh lod = build_lod(m, false);
    const geometry g = pack(lod);
    CHECK(g.leaf_triangles() == m.triangle_count());
    CHECK(g.textured() == m.textured());
    if (g.textured()) {
        // every wedge has its vertex's coordinates in its chart (unless seams moved), and every
        // triangle one chart. only the last group's vertex clustering may smear one.
        size_t off = 0, mixed = 0, mixed_root = 0;
        for (const gpu_cluster& c : g.clusters) {
            for (uint32_t k = 0; k < c.vertex_count; ++k) {
                const uint32_t w = g.cluster_vertices[c.vertex_offset + k];
                const vec2 uv = g.wedge_uvs[w];
                const vec2 want = chart_uv(unplaced[g.vertex_of(w)], chart_from_u(uv.x));
                off += std::abs(uv.x - want.x) > 1e-6f || std::abs(uv.y - want.y) > 1e-6f;
            }
            for (uint32_t t = 0; t < c.triangle_count; ++t) {
                const uint32_t p = g.cluster_triangles[c.triangle_offset + t];
                const uint32_t* v = &g.cluster_vertices[c.vertex_offset];
                const uint32_t a = chart_from_u(g.wedge_uvs[v[p & 255]].x), b = chart_from_u(g.wedge_uvs[v[p >> 8 & 255]].x),
                               d = chart_from_u(g.wedge_uvs[v[p >> 16 & 255]].x);
                const bool bad = a != b || b != d;
                mixed += bad;
                mixed_root += bad && c.parent_error >= 1e30f;
            }
        }
        std::printf("  %zu wedges; %zu off their vertex's coordinates; %zu triangles across charts (%zu in roots)\n",
                    g.wedge_uvs.size(), off, mixed, mixed_root);
        CHECK(!exact || off == 0);
        CHECK(mixed == mixed_root);
    }
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
    CHECK(worst_area > min_area);

    // paged, written, read back: every cluster's vertices and triangles come
    // out of its page as they went in; dependencies are coarser pages.
    const paged_geometry paged = page(g);
    const std::string path = temp_path("test.cgeo");
    save_paged(paged, path);
    const paged_geometry back = load_paged(path, true);
    CHECK(back.clusters.size() == g.clusters.size() && back.pages.size() == paged.pages.size());
    // cut short anywhere around the padding before the page data, it fails to load (once it
    // spun forever: past the end, tellg is -1, and -1 % 16 is never 0).
    {
        const std::string cut = temp_path("cut.cgeo");
        size_t refused = 0, tried = 0;
        for (uint64_t at = back.data_offset - 24; at < back.data_offset + 4; ++at, ++tried) {
            std::filesystem::copy_file(path, cut, std::filesystem::copy_options::overwrite_existing);
            std::filesystem::resize_file(cut, at);
            try {
                load_paged(cut, true);
            } catch (const std::runtime_error&) {
                ++refused;
            }
        }
        CHECK(refused == tried);
    }
    // positions within half a grid step, normals within a fraction of a degree,
    // shared vertices identical from every cluster: quantizing opens no cracks.
    size_t wrong = 0, mismatched = 0;
    float worst_normal = 1, worst_position = 0, worst_uv = 0;
    std::map<uint32_t, vec3> seen;
    for (size_t i = 0; i < g.clusters.size(); ++i) {
        const gpu_cluster& c = g.clusters[i];
        const gpu_cluster& pc = back.clusters[i];
        const uint32_t* base = &back.data[back.pages[pc.group].offset / 4];
        for (uint32_t k = 0; k < c.vertex_count; ++k) {
            const uint32_t w = g.cluster_vertices[c.vertex_offset + k], v = g.vertex_of(w);
            if (g.textured()) {
                const vec2 uv = decode_uv(pc, base, k), want = g.wedge_uvs[w];
                worst_uv = std::max({worst_uv, std::abs(uv.x - want.x), std::abs(uv.y - want.y)});
            }
            const vec3 p = decode_position(back, pc, base, k);
            const vec3 original(g.positions[3 * v], g.positions[3 * v + 1], g.positions[3 * v + 2]);
            worst_position = std::max(worst_position, length(p - original));
            auto [it, added] = seen.emplace(v, p);
            if (!added) mismatched += std::memcmp(&it->second, &p, sizeof p) != 0;
            const vec3 n = decode_normal(pc, base, k);
            worst_normal = std::min(worst_normal, dot(n, vec3(g.normals[3 * v], g.normals[3 * v + 1], g.normals[3 * v + 2])));
        }
        for (uint32_t t = 0; t < c.triangle_count; ++t) wrong += decode_triangle(pc, base, t) != g.cluster_triangles[c.triangle_offset + t];
    }
    std::printf("  %zu pages, %.0f KB; positions within %.2f grid steps, normals within %.3f degrees\n", back.pages.size(),
                back.data_size / 1024.0, worst_position / back.grid_step, std::acos(std::min(1.0f, worst_normal)) * 57.2958f);
    CHECK(wrong == 0);
    CHECK(mismatched == 0);
    CHECK(worst_position <= 0.88f * back.grid_step);  // half a step per axis
    CHECK(worst_normal > 0.9999f);
    CHECK(worst_uv <= 0.5f / 65535 + 1e-7f);
    if (g.textured()) std::printf("  texture coordinates within %.2f steps of 1/65535\n", worst_uv * 65535);
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

// bc1 blocks within a few levels on smooth colour, and a texture file's tiles: each texel of
// a tile, border included, is its level's texel (clamped at the edges), through bc1.
void test_texture() {
    std::printf("textures\n");
    uint8_t rgb[48], back[48], block[8];
    int worst_smooth = 0;
    for (int b = 0; b < 200; ++b) {
        for (int i = 0; i < 16; ++i) {
            const float t = (i % 4 + i / 4) / 6.0f;
            rgb[3 * i] = uint8_t(30 + b + 20 * t);
            rgb[3 * i + 1] = uint8_t(200 - b / 2 - 30 * t);
            rgb[3 * i + 2] = uint8_t(90 + 10 * t);
        }
        encode_bc1(rgb, block);
        decode_bc1(block, back);
        for (int i = 0; i < 48; ++i) worst_smooth = std::max(worst_smooth, std::abs(rgb[i] - back[i]));
    }
    std::printf("  bc1 on gradients: within %d of 255\n", worst_smooth);
    CHECK(worst_smooth <= 10);

    image im;
    im.width = 300;
    im.height = 130;
    for (uint32_t y = 0; y < im.height; ++y)
        for (uint32_t x = 0; x < im.width; ++x)
            im.rgb.insert(im.rgb.end(), {uint8_t(x * 255 / 299), uint8_t(y * 255 / 129), uint8_t(128 + 100 * std::sin(x * 0.05f))});
    const std::string path = temp_path("test.ctex");
    save_texture(im, path);
    const texture_info t = load_texture_info(path);
    CHECK(t.textures.size() == 1);
    const std::vector<texture_level>& lv = t.textures[0].levels;
    CHECK(lv.size() == 3);  // 300 x 130, 150 x 65, 75 x 33
    CHECK(lv.size() == 3 && lv[0].tiles_x == 3 && lv[0].tiles_y == 2 && lv[2].tiles_x == 1);
    CHECK(t.tile_count() == 6 + 2 + 1);
    CHECK(t.parent(4) == 6);  // (1, 1) of level 0 -> (0, 0) of level 1
    CHECK(t.parent(5) == 7);  // (2, 1) -> (1, 0)
    CHECK(t.parent(8) == UINT32_MAX);
    // level 0 tile (2, 1): its texels against the image.
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> tile(tile_bytes);
    f.seekg(static_cast<std::streamoff>(t.data_offset + 5 * tile_bytes));
    f.read(reinterpret_cast<char*>(tile.data()), tile_bytes);
    int worst = 0;
    for (uint32_t by = 0; by < tile_texels / 4; ++by)
        for (uint32_t bx = 0; bx < tile_texels / 4; ++bx) {
            decode_bc1(&tile[(by * 32 + bx) * 8], back);
            for (int i = 0; i < 16; ++i) {
                const int x = std::clamp(int(2 * tile_payload) - int(tile_border) + int(4 * bx) + i % 4, 0, 299);
                const int y = std::clamp(int(tile_payload) - int(tile_border) + int(4 * by) + i / 4, 0, 129);
                for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(back[3 * i + c] - im.rgb[(size_t(y) * 300 + x) * 3 + c]));
            }
        }
    std::printf("  a tile against its image: within %d of 255\n", worst);
    CHECK(worst <= 12);
}

std::string base64_of(const std::string& bytes) {
    static const char* d = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t v = uint8_t(bytes[i]) << 16 | (i + 1 < bytes.size() ? uint8_t(bytes[i + 1]) << 8 : 0) |
                     (i + 2 < bytes.size() ? uint8_t(bytes[i + 2]) : 0);
        out += d[v >> 18 & 63];
        out += d[v >> 12 & 63];
        out += i + 1 < bytes.size() ? d[v >> 6 & 63] : '=';
        out += i + 2 < bytes.size() ? d[v & 63] : '=';
    }
    return out;
}

// a skinned gltf: a bar along x from 0 to 2, two triangles a unit square, joint 0 at the
// origin and joint 1 (its child) at x = 1; points left of 1 follow joint 0, right of it
// joint 1, those at 1 both. joint 1 turns a quarter about z over one second.
std::string bent_bar_gltf() {
    std::vector<float> pos, wts;
    std::vector<uint16_t> jts;
    std::vector<uint16_t> idx;
    for (int x = 0; x <= 2; ++x)
        for (int y = 0; y <= 1; ++y) {
            pos.insert(pos.end(), {float(x), float(y), 0});
            jts.insert(jts.end(), {0, 1, 0, 0});
            const float w1 = x == 0 ? 0.0f : x == 1 ? 0.5f : 1.0f;
            wts.insert(wts.end(), {1 - w1, w1, 0, 0});
        }
    for (uint16_t x = 0; x < 2; ++x) {
        const uint16_t a = 2 * x, b = a + 1, c = a + 2, e = a + 3;
        idx.insert(idx.end(), {a, c, b, b, c, e});
    }
    const float ibm[32] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                           1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1};
    const float times[2] = {0, 1};
    const float rot[8] = {0, 0, 0, 1, 0, 0, 0.70710678f, 0.70710678f};
    std::string bin;
    auto add = [&](const void* p, size_t n) { size_t at = bin.size(); bin.append(static_cast<const char*>(p), n); while (bin.size() % 4) bin += char(0); return at; };
    const size_t o_pos = add(pos.data(), pos.size() * 4), o_jts = add(jts.data(), jts.size() * 2), o_wts = add(wts.data(), wts.size() * 4),
                 o_idx = add(idx.data(), idx.size() * 2), o_ibm = add(ibm, sizeof ibm), o_t = add(times, sizeof times), o_r = add(rot, sizeof rot);
    char json[4096];
    std::snprintf(json, sizeof json, R"({
  "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 2]}],
  "nodes": [{"name": "j0", "children": [1]}, {"name": "j1", "translation": [1, 0, 0]}, {"mesh": 0, "skin": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}, "indices": 3}]}],
  "skins": [{"joints": [0, 1], "inverseBindMatrices": 4}],
  "animations": [{"name": "bend", "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}],
                  "samplers": [{"input": 5, "output": 6}]}],
  "buffers": [{"byteLength": %zu, "uri": "data:application/octet-stream;base64,%s"}],
  "bufferViews": [{"buffer": 0, "byteOffset": %zu, "byteLength": 72}, {"buffer": 0, "byteOffset": %zu, "byteLength": 48},
                  {"buffer": 0, "byteOffset": %zu, "byteLength": 96}, {"buffer": 0, "byteOffset": %zu, "byteLength": 24},
                  {"buffer": 0, "byteOffset": %zu, "byteLength": 128}, {"buffer": 0, "byteOffset": %zu, "byteLength": 8},
                  {"buffer": 0, "byteOffset": %zu, "byteLength": 32}],
  "accessors": [{"bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": 6, "type": "VEC4"},
                {"bufferView": 2, "componentType": 5126, "count": 6, "type": "VEC4"},
                {"bufferView": 3, "componentType": 5123, "count": 12, "type": "SCALAR"},
                {"bufferView": 4, "componentType": 5126, "count": 2, "type": "MAT4"},
                {"bufferView": 5, "componentType": 5126, "count": 2, "type": "SCALAR"},
                {"bufferView": 6, "componentType": 5126, "count": 2, "type": "VEC4"}]
})", bin.size(), base64_of(bin).c_str(), o_pos, o_jts, o_wts, o_idx, o_ibm, o_t, o_r);
    return json;
}

vec3 apply_rows(const float* r, vec3 p) {
    return {r[0] * p.x + r[1] * p.y + r[2] * p.z + r[3], r[4] * p.x + r[5] * p.y + r[6] * p.z + r[7],
            r[8] * p.x + r[9] * p.y + r[10] * p.z + r[11]};
}

vec3 skinned(const mesh& m, const std::vector<float>& rows, size_t v) {
    vec3 out(0, 0, 0);
    for (int k = 0; k < 4; ++k)
        out += apply_rows(&rows[12 * m.skin_joints[4 * v + k]], m.positions[v]) * (m.skin_weights[4 * v + k] / 255.0f);
    return out;
}

// gltf reading and posing: the rest pose is the identity, joint 1's quarter turn bends the
// bar's right end up, the same once the model is placed (joints conjugated), and the skin
// survives subdivision, the hierarchy and pages, its page bounds only growing to the roots.
void test_skinning() {
    std::printf("skinning\n");
    const std::string path = temp_path("bar.gltf");
    std::ofstream(path) << bent_bar_gltf();
    skeleton sk;
    mesh m = load_mesh(path, &sk);
    CHECK(m.skinned() && m.positions.size() == 6 && m.triangle_count() == 4);
    CHECK(sk.joints.size() == 2 && sk.animations.size() == 1 && sk.animations[0].duration == 1);
    std::vector<float> rows;
    pose_joints(sk, -1, 0, rows);
    float worst = 0;
    for (size_t v = 0; v < m.positions.size(); ++v) worst = std::max(worst, length(skinned(m, rows, v) - m.positions[v]));
    CHECK(worst < 1e-5f);
    pose_joints(sk, 0, 1 - 1e-6f, rows);  // the turn's end (time wraps at the duration)
    // (2, 0) follows joint 1 alone: turned a quarter about (1, 0) to (1, 1); (1, y) stays.
    CHECK(length(skinned(m, rows, 4) - vec3(1, 1, 0)) < 1e-3f);
    CHECK(length(skinned(m, rows, 2) - m.positions[2]) < 1e-3f);

    weld(m);
    for (int k = 0; k < 3; ++k) subdivide(m);
    CHECK(m.skinned() && m.skin_weights.size() == 4 * m.positions.size());
    bool sums = true;
    for (size_t v = 0; v < m.positions.size(); ++v)
        sums &= m.skin_weights[4 * v] + m.skin_weights[4 * v + 1] + m.skin_weights[4 * v + 2] + m.skin_weights[4 * v + 3] == 255;
    CHECK(sums);
    // placed: every vertex posed then placed is the placed vertex posed with conjugated joints.
    pose_joints(sk, 0, 0.6f, rows);
    std::vector<vec3> posed(m.positions.size());
    for (size_t v = 0; v < posed.size(); ++v) posed[v] = skinned(m, rows, v);
    sk.place = normalize_placement(m, false);
    pose_joints(sk, 0, 0.6f, rows);
    float placed_worst = 0;
    for (size_t v = 0; v < posed.size(); ++v)
        placed_worst = std::max(placed_worst, length(skinned(m, rows, v) - (posed[v] - sk.place.base) * sk.place.scale));
    CHECK(placed_worst < 1e-5f);

    const lod_mesh lod = build_lod(m, false);
    const geometry g = pack(lod);
    const paged_geometry pg = page(g);
    CHECK(pg.skin.size() == pg.pages.size());
    size_t shrinking = 0, wrong = 0;
    for (const gpu_cluster& c : pg.clusters) {
        if (c.creator == no_page) continue;
        const page_skin &fine = pg.skin[c.creator], &coarse = pg.skin[c.group];
        shrinking += (fine.joints & ~coarse.joints) != 0;
    }
    // every vertex's joints and weights out of its page as they went in.
    std::map<uint32_t, uint32_t> page_start;
    for (size_t i = 0; i < pg.clusters.size(); ++i) {
        const gpu_cluster& c = g.clusters[i];
        const uint32_t* base = &pg.data[pg.pages[pg.clusters[i].group].offset / 4];
        for (uint32_t k = 0; k < c.vertex_count; ++k) {
            uint32_t j, w;
            decode_skin(pg.clusters[i], base, k, j, w);
            const size_t v = 4 * size_t(g.vertex_of(g.cluster_vertices[c.vertex_offset + k]));
            for (int a = 0; a < 4; ++a) wrong += (j >> (8 * a) & 255) != g.skin_joints[v + a] || (w >> (8 * a) & 255) != g.skin_weights[v + a];
        }
        // its page's joints cover its vertices.
        for (uint32_t k = 0; k < c.vertex_count; ++k) {
            const size_t v = 4 * size_t(g.vertex_of(g.cluster_vertices[c.vertex_offset + k]));
            for (int a = 0; a < 4; ++a)
                if (g.skin_weights[v + a]) wrong += !(pg.skin[pg.clusters[i].group].joints >> g.skin_joints[v + a] & 1);
        }
    }
    std::printf("  %zu triangles, %zu pages; skin bounds shrinking toward the roots: %zu, skin data wrong: %zu\n",
                m.triangle_count(), pg.pages.size(), shrinking, wrong);
    CHECK(shrinking == 0);
    CHECK(wrong == 0);
    // the file round trip.
    sk.pages = pg.skin;
    save_skeleton(sk, temp_path("bar.cskn"));
    const skeleton back = load_skeleton(temp_path("bar.cskn"));
    CHECK(back.joints == sk.joints && back.pages.size() == sk.pages.size() && back.animations.size() == 1 &&
          back.animations[0].channels[0].values == sk.animations[0].channels[0].values);
}

// two materials and repeating coordinates: a grid, its left half one material and its right
// the other (a seam down the middle), coordinates four and three repeats across. every vertex
// decodes back to its coordinates (within a step of the range) and material, and no triangle
// has two; a repeating texture's tile borders wrap.
void test_materials() {
    std::printf("materials\n");
    mesh m = grid(60);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> wedge_of;
    for (size_t t = 0; t < m.triangle_count(); ++t) {
        const vec3 c = (m.positions[m.indices[3 * t]] + m.positions[m.indices[3 * t + 1]] + m.positions[m.indices[3 * t + 2]]) * (1.0f / 3);
        const uint32_t mat = c.x > 0.5f ? 1 : 0;
        for (int k = 0; k < 3; ++k) {
            const uint32_t v = m.indices[3 * t + k];
            auto [it, added] = wedge_of.emplace(std::make_pair(v, mat), uint32_t(m.wedge_uvs.size()));
            if (added) {
                m.wedge_uvs.push_back({4 * m.positions[v].x, 3 * m.positions[v].z - 1});
                m.wedge_vertex.push_back(v);
                m.wedge_material.push_back(uint8_t(mat));
            }
            m.corners.push_back(it->second);
        }
    }
    m.materials = {material{}, material{}};
    const lod_mesh lod = build_lod(m, false);
    const geometry g = pack(lod);
    const paged_geometry pg = page(g);
    CHECK(g.uv_min.x == 0 && g.uv_min.y == -1 && g.uv_extent == 4);
    size_t wrong_uv = 0, wrong_material = 0, mixed = 0;
    for (size_t i = 0; i < g.clusters.size(); ++i) {
        const gpu_cluster& c = g.clusters[i];
        const uint32_t* base = &pg.data[pg.pages[pg.clusters[i].group].offset / 4];
        uint32_t mats[cluster_max_vertices];
        for (uint32_t k = 0; k < c.vertex_count; ++k) {
            const uint32_t w = g.cluster_vertices[c.vertex_offset + k];
            const vec2 q = decode_uv(pg.clusters[i], base, k, &mats[k]);
            const vec2 uv = {g.uv_min.x + q.x * g.uv_extent, g.uv_min.y + q.y * g.uv_extent};
            wrong_uv += std::abs(uv.x - g.wedge_uvs[w].x) > g.uv_extent / 65535 || std::abs(uv.y - g.wedge_uvs[w].y) > g.uv_extent / 65535;
            wrong_material += mats[k] != g.wedge_material[w];
        }
        for (uint32_t t = 0; t < c.triangle_count; ++t) {
            const uint32_t p = g.cluster_triangles[c.triangle_offset + t];
            mixed += mats[p & 255] != mats[p >> 8 & 255] || mats[p & 255] != mats[p >> 16 & 255];
        }
    }
    std::printf("  %zu clusters; coordinates off %zu, materials off %zu, triangles of two materials %zu\n", g.clusters.size(),
                wrong_uv, wrong_material, mixed);
    CHECK(wrong_uv == 0 && wrong_material == 0 && mixed == 0);

    // a repeating 200 x 50 texture: tile (0, 0)'s left border is the image's right edge.
    image im;
    im.width = 200;
    im.height = 50;
    for (uint32_t y = 0; y < 50; ++y)
        for (uint32_t x = 0; x < 200; ++x) im.rgb.insert(im.rgb.end(), {uint8_t(x), uint8_t(4 * y), uint8_t(255 - x)});
    const std::string path = temp_path("wrap.ctex");
    save_textures({im}, {1}, {0, 0}, 1, path);
    const texture_info t = load_texture_info(path);
    CHECK(t.textures.size() == 1 && t.textures[0].repeat);
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> tile(tile_bytes);
    f.seekg(static_cast<std::streamoff>(t.data_offset));
    f.read(reinterpret_cast<char*>(tile.data()), tile_bytes);
    uint8_t back[48];
    int worst = 0;
    for (uint32_t by = 1; by < 4; ++by) {  // the border block column, rows inside the image
        decode_bc1(&tile[(by * 32) * 8], back);
        for (int i = 0; i < 16; ++i) {
            const int x = 200 - 4 + i % 4, y = int(4 * by) - 4 + i / 4;  // wrapped from the right edge
            for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(back[3 * i + c] - im.rgb[(size_t(y) * 200 + x) * 3 + c]));
        }
    }
    std::printf("  a repeating texture's border against the far edge: within %d of 255\n", worst);
    CHECK(worst <= 12);
}

}  // namespace

// packed normal cones (packed_cluster) never cull what the exact cone keeps: the culling test
// dot(v, axis) >= cutoff * |v| + r, for random cones and views.
void test_packed_cones() {
    std::printf("packed cones\n");
    uint32_t seed = 1;
    auto rand = [&] {
        seed = seed * 1664525u + 1013904223u;
        return float(seed >> 8) / float(1u << 24);
    };
    auto direction = [&] {
        for (;;) {
            const vec3 v(rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1);
            if (length(v) > 0.1f && length(v) <= 1) return normalize(v);
        }
    };
    const std::vector<page_bounds> shared = {{{0, 0, 0}, 1, 0}};
    int wrong = 0, culled = 0;
    for (int k = 0; k < 2000; ++k) {
        gpu_cluster c{};
        const vec3 axis = direction();
        c.cone_axis[0] = axis.x;
        c.cone_axis[1] = axis.y;
        c.cone_axis[2] = axis.z;
        c.cone_cutoff = rand() * 2 - 1;
        c.creator = no_page;
        c.vertex_count = c.triangle_count = 128;
        const gpu_cluster u = unpack_cluster(pack_cluster(c), shared);
        const vec3 packed_axis(u.cone_axis[0], u.cone_axis[1], u.cone_axis[2]);
        for (int j = 0; j < 200; ++j) {
            const vec3 v = direction() * (0.5f + rand() * 10);
            const float r = rand() * 0.5f;
            if (u.cone_cutoff < 1 && dot(v, packed_axis) >= u.cone_cutoff * length(v) + r) {
                ++culled;
                if (dot(v, axis) < c.cone_cutoff * length(v) + r) ++wrong;
            }
        }
    }
    std::printf("  %d views culled, %d of them wrongly\n", culled, wrong);
    CHECK(culled > 0);
    CHECK(wrong == 0);
}

// the deviation bound holds: at least the largest distance found by brute force from
// thousands of random points on either mesh to the other, and close to it.
void test_deviation_bound() {
    std::printf("deviation bound\n");
    const mesh m = bumpy_sphere(4);  // 2048 triangles
    const std::vector<uint8_t> locked(m.positions.size(), 0);
    const simplify_result r = simplify(m.positions, m.indices, locked, 300);
    uint32_t seed = 7;
    auto rand = [&] {
        seed = seed * 1664525u + 1013904223u;
        return float(seed >> 8) / float(1u << 24);
    };
    auto farthest = [&](const std::vector<uint32_t>& from, const std::vector<uint32_t>& to) {
        float worst = 0;
        for (size_t t = 0; t < from.size(); t += 3)
            for (int k = 0; k < 40; ++k) {
                float u = rand(), v = rand();
                if (u + v > 1) { u = 1 - u; v = 1 - v; }
                const vec3 a = m.positions[from[t]], b = m.positions[from[t + 1]], c = m.positions[from[t + 2]];
                const vec3 p = a + (b - a) * u + (c - a) * v;
                float best = INFINITY;
                for (size_t s = 0; s < to.size(); s += 3)
                    best = std::min(best, point_triangle_distance(p, m.positions[to[s]], m.positions[to[s + 1]], m.positions[to[s + 2]]));
                worst = std::max(worst, best);
            }
        return worst;
    };
    const float brute = std::max(farthest(m.indices, r.indices), farthest(r.indices, m.indices));
    const float sampled = sampled_deviation(m.positions, m.indices, r.indices);
    const float bound = mesh_deviation(m.positions, m.indices, r.indices);
    std::printf("  sampled %.6f, brute force %.6f, bound %.6f\n", sampled, brute, bound);
    CHECK(bound >= brute);
    CHECK(bound >= sampled);
    CHECK(bound <= 1.25f * std::max(brute, sampled));
}

// foliage keeps its coverage (dag.cpp): grow_to_area() gives clustered pieces back the area
// of what merged into them without moving a locked vertex, and every foliage group's error is
// at least its inputs' plus the side of the square of area it lost.
void test_foliage_coverage() {
    std::printf("foliage coverage\n");
    auto area = [](const std::vector<vec3>& p, const std::vector<uint32_t>& idx) {
        double a = 0;
        for (size_t t = 0; t < idx.size(); t += 3) a += 0.5 * length(cross(p[idx[t + 1]] - p[idx[t]], p[idx[t + 2]] - p[idx[t]]));
        return a;
    };

    // 400 thin cards in a small ball, clustered hard, one vertex locked.
    mesh m = islands(400, 3, true);
    for (vec3& p : m.positions) p = p * 0.3f;
    std::vector<uint8_t> locked(m.positions.size(), 0);
    locked[m.indices[0]] = 1;
    simplify_result r = cluster_vertices(m.positions, m.indices, 60, nullptr, &locked);
    const double before = area(m.positions, m.indices), clustered = area(m.positions, r.indices);
    grow_to_area(m.positions, m.indices, r, locked, nullptr, 3.0f);
    std::vector<vec3> all = m.positions;
    all.insert(all.end(), r.new_positions.begin(), r.new_positions.end());
    std::vector<uint32_t> grown = r.indices;
    for (uint32_t& v : grown)
        if (v & new_vertex_bit) v = uint32_t(m.positions.size()) + (v & ~new_vertex_bit);
    const double after = area(all, grown);
    std::printf("  area: %.4f in, %.4f clustered, %.4f grown\n", before, clustered, after);
    CHECK(clustered < before * 0.9);  // the clustering lost area (else the test tests nothing)
    CHECK(after > clustered * 1.2 && after <= before * 1.05);
    for (uint32_t k = 0; k < r.new_from.size(); ++k) CHECK(!locked[r.new_from[k]]);
    for (uint32_t v : r.indices) CHECK(!(v & new_vertex_bit) || !locked[r.new_from[v & ~new_vertex_bit]]);

    // the error's area term, over a whole hierarchy of cards.
    m = islands(20000, 9, true);
    weld(m);
    const lod_mesh lod = build_lod(m, false);
    std::map<uint32_t, std::vector<const lod_cluster*>> inputs, outputs;
    for (const lod_cluster& c : lod.clusters) {
        if (c.group != UINT32_MAX) inputs[c.group].push_back(&c);
        if (c.creator != UINT32_MAX) outputs[c.creator].push_back(&c);
    }
    size_t groups = 0, short_of = 0;
    for (const auto& [g, in] : inputs) {
        if (!outputs.count(g)) continue;
        std::vector<uint32_t> in_idx, out_idx;
        float child = 0;
        for (const lod_cluster* c : in) {
            in_idx.insert(in_idx.end(), c->indices.begin(), c->indices.end());
            child = std::max(child, c->lod_error);
        }
        float error = INFINITY;
        for (const lod_cluster* c : outputs[g]) {
            out_idx.insert(out_idx.end(), c->indices.begin(), c->indices.end());
            error = std::min(error, c->lod_error);
        }
        const double lost = std::max(0.0, area(lod.positions, in_idx) - area(lod.positions, out_idx));
        ++groups;
        if (error < (child + std::sqrt(lost)) * 0.999 - 1e-7) ++short_of;
    }
    std::printf("  %zu groups, %zu with an error short of the area they lost\n", groups, short_of);
    CHECK(groups > 100 && short_of == 0);
}

int main() {
    test_foliage_coverage();
    test_materials();
    test_skinning();
    test_texture();
    test_deviation_bound();
    test_packed_cones();
    test_ply_and_obj();
    test_weld();
    test_trim_empty();
    test_clusters();
    test_simplify();
    test_hierarchy("closed sphere", bumpy_sphere(7));
    test_hierarchy("sphere with holes", holed_sphere(7));
    test_hierarchy("flat grid", grid(250));
    test_hierarchy("textured sphere with holes", grid_textured(holed_sphere(7), 2));
    test_hierarchy("foliage: 30000 islands", islands(30000, 5), true, 0.15);
    texture_charts = 216;
    test_hierarchy("sphere with holes, 216 small charts", grid_textured(holed_sphere(7), 6), false);
    if (failures) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
}
