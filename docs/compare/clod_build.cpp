// builds a colossus .cgeo with meshoptimizer's clusterlod (demo/clusterlod.h) in place of
// colossus's own builder, so the same renderer, cameras and measurements compare the two:
// clod_build in.ply out.cgeo [--up-z] [--check]. `make compare` builds it against a meshoptimizer
// clone beside this one (MESHOPT=path to change); docs/compare/compare.py measures.
#define CLUSTERLOD_IMPLEMENTATION
#include "meshoptimizer.h"
#include "clusterlod.h"

#include "dag.hpp"
#include "geometry_file.hpp"
#include "lod_check.hpp"
#include "mesh.hpp"
#include "paged_file.hpp"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) return std::fprintf(stderr, "clod_build in out.cgeo [--up-z] [--check]\n"), 1;
    bool up_z = false, check = false;
    for (int i = 3; i < argc; ++i) {
        up_z |= !std::strcmp(argv[i], "--up-z");
        check |= !std::strcmp(argv[i], "--check");
    }
    const auto start = std::chrono::steady_clock::now();
    auto since = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };

    // loaded, welded and placed as colossus_build does.
    mesh m = load_mesh(argv[1]);
    weld(m);
    normalize_placement(m, up_z);
    const std::vector<vec3> normals = vertex_normals(m);
    std::printf("read %s: %zu vertices, %zu triangles (%.1fs)\n", argv[1], m.positions.size(), m.triangle_count(), since());

    // meshoptimizer's own defaults, as its nanite demo uses them: 128 triangles and vertices a
    // cluster, normals as attributes at weight 0.5.
    std::vector<float> attributes(normals.size() * 3);
    for (size_t i = 0; i < normals.size(); ++i) std::memcpy(&attributes[3 * i], &normals[i], 12);
    const float weights[3] = {0.5f, 0.5f, 0.5f};
    clodConfig config = clodDefaultConfig(128);
    clodMesh cm = {};
    cm.indices = m.indices.data();
    cm.index_count = m.indices.size();
    cm.vertex_count = m.positions.size();
    cm.vertex_positions = &m.positions[0].x;
    cm.vertex_positions_stride = sizeof(vec3);
    cm.vertex_attributes = attributes.data();
    cm.vertex_attributes_stride = 12;
    cm.attribute_weights = weights;
    cm.attribute_count = 3;

    // a clusterlod group is a colossus group: its clusters were simplified together into
    // `simplified`, their parent; each cluster's own error is that of the group it came from
    // (refined), or none for a leaf. a terminal group (error FLT_MAX) holds roots.
    lod_mesh lod;
    lod.positions = m.positions;
    lod.normals = normals;
    std::vector<clodBounds> simplified;
    std::vector<int> depth;
    std::vector<lod_cluster> clusters;
    clodBuild(config, cm, [&](clodGroup group, const clodCluster* cs, size_t count) -> int {
        const int id = int(simplified.size());
        simplified.push_back(group.simplified);
        depth.push_back(group.depth);
        const bool root = group.simplified.error == FLT_MAX;
        for (size_t i = 0; i < count; ++i) {
            lod_cluster c;
            c.indices.assign(cs[i].indices, cs[i].indices + cs[i].index_count);
            c.corners = c.indices;
            cluster_bounds(lod.positions, c);
            c.level = uint32_t(group.depth);
            c.group = root ? UINT32_MAX : uint32_t(id);
            c.creator = cs[i].refined < 0 ? UINT32_MAX : uint32_t(cs[i].refined);
            if (cs[i].refined < 0) {
                c.lod_bounds = c.bounds;
                c.lod_error = 0;
            } else {
                const clodBounds& b = simplified[size_t(cs[i].refined)];
                c.lod_bounds = {{b.center[0], b.center[1], b.center[2]}, b.radius};
                c.lod_error = b.error;
            }
            if (root) {
                c.parent_bounds = c.lod_bounds;
                c.parent_error = INFINITY;
            } else {
                c.parent_bounds = {{group.simplified.center[0], group.simplified.center[1], group.simplified.center[2]}, group.simplified.radius};
                c.parent_error = group.simplified.error;
            }
            clusters.push_back(std::move(c));
        }
        return id;
    });
    std::stable_sort(clusters.begin(), clusters.end(), [](const lod_cluster& a, const lod_cluster& b) { return a.level < b.level; });
    lod.clusters = std::move(clusters);
    for (const lod_cluster& c : lod.clusters) {
        if (lod.levels.size() <= c.level) lod.levels.resize(c.level + 1);
        lod_level_stats& s = lod.levels[c.level];
        ++s.clusters;
        s.triangles += c.indices.size() / 3;
        s.max_error = std::max(s.max_error, c.lod_error);
    }
    for (size_t g = 0; g < depth.size(); ++g) ++lod.levels[size_t(depth[g])].groups;
    std::printf("clusterlod: %zu clusters, %zu groups, %zu levels (%.1fs)\n", lod.clusters.size(), simplified.size(), lod.levels.size(), since());

    const geometry g = pack(lod);
    if (check) {
        size_t cracked = 0;
        for (const cut_report& r : check_cuts(g, 24)) cracked += r.cracked_edges;
        std::printf("cracked edges over 25 cuts: %zu\n", cracked);
    }
    const paged_geometry paged = page(g);
    save_paged(paged, argv[2]);
    std::printf("wrote %s: %zu clusters, %zu pages, %.0f MB (%.1fs)\n", argv[2], paged.clusters.size(), paged.pages.size(),
                paged.data_size / 1048576.0, since());
}
