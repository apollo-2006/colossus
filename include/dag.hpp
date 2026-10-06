#pragma once
// the cluster lod hierarchy: a dag where each level is a coarser copy of the
// one below, and any mix of levels joins without cracks.
//
// per level:
//   1. group clusters, about eight, by shared edges.
//   2. lock vertices shared between groups; merge each group and simplify to
//      half.
//   3. split the result into clusters: the next level.
// locked outlines meet edge for edge simplified or not. groups differ per
// level, so no seam outlives a level.
//
// each cluster stores its own error and its parent group's, each with a sphere,
// equal across a group and growing toward the root, so
//
//     own error on screen <= threshold < parent error on screen
//
// picks exactly one level on every leaf to root path, and a group agrees. one
// gpu thread per cluster, no traversal.
#include "mesh.hpp"

#include <cmath>
#include <vector>

struct lod_cluster {
    std::vector<uint32_t> indices;  // three per triangle, into lod_mesh::positions
    std::vector<uint32_t> corners;  // a wedge per index (lod_mesh::wedge_*)
    sphere bounds;                  // around the vertices: for culling
    vec3 cone_axis;                 // holds every face normal: see cone_cutoff
    // sine of the widest angle from cone_axis to a face normal: all triangles
    // face away from direction d (from the bounds' centre) when dot(d, axis) >=
    // cone_cutoff + radius / distance. 1 disables.
    float cone_cutoff = 1;

    sphere lod_bounds;  // same for every cluster of a group
    float lod_error = 0;
    sphere parent_bounds;              // the group it was simplified in
    float parent_error = INFINITY;    // infinite for a root
    uint32_t level = 0;
    uint32_t group = UINT32_MAX;       // group it was simplified in, if any
    // group whose simplification made it: the finer clusters it stands for.
    // none for leaves.
    uint32_t creator = UINT32_MAX;
};

struct lod_level_stats {
    size_t clusters = 0, triangles = 0, groups = 0, stuck_groups = 0;
    float max_error = 0;
    uint32_t unused = 0;  // files store this struct: no stray padding bytes
};
static_assert(sizeof(lod_level_stats) == 40, "web/split.py reads 40 bytes a level");

// texture coordinates ride on wedges: a vertex as one texture chart sees it. simplification
// keeps them exact (a corner only ever takes a wedge of its own chart: see simplify()), so
// seams stay closed. an untextured mesh has a wedge per vertex, in one chart.
struct lod_mesh {
    std::vector<vec3> positions, normals;
    std::vector<uint32_t> wedge_vertex, wedge_chart;
    std::vector<vec2> wedge_uvs;  // empty if untextured
    std::vector<uint8_t> wedge_material;
    std::vector<material> materials;
    std::vector<uint8_t> skin_joints, skin_weights;  // per vertex, empty if unskinned (mesh.hpp)
    // per vertex, the vertex it was grown from (itself if not grown: grow_to_area()); empty if
    // none was. the crack check follows it to the original.
    std::vector<uint32_t> vertex_source;
    std::vector<lod_cluster> clusters;  // level 0, then each level
    std::vector<lod_level_stats> levels;
};

// builds the hierarchy over a welded mesh. one log line per level if verbose.
//
// skinned, with poses (joint matrices, 3x4 rows each, in the mesh's space): each group's
// simplification is also measured posed, every pose (sampled distance between the posed
// surfaces), and the worst joins its error. linear blend skinning moves a coarse triangle off
// the fine surface where its interpolated weights differ, by however far apart the joints
// carry a point: the animations are known, so measured over them rather than bounded.
//
// normal_weight > 0 also prices collapses by the change of normal (simplify()).
lod_mesh build_lod(const mesh& m, bool verbose, const std::vector<std::vector<float>>& poses = {}, float normal_weight = 0);

// cone and bounds of a cluster.
void cluster_bounds(const std::vector<vec3>& positions, lod_cluster& c);
