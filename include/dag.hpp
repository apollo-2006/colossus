#pragma once
// The cluster LOD hierarchy: a DAG of clusters in which every level is a
// coarser copy of the one below, built so that any mix of levels joins up
// without cracks.
//
// Building it, level by level:
//   1. Partition the level's clusters into groups of about eight that share
//      many edges.
//   2. Lock every vertex a group shares with another group, merge each
//      group's triangles and simplify them to half.
//   3. Split the simplified triangles into new clusters: the next level.
// Because group outlines are locked, a group simplified and a neighbouring
// group not simplified still meet edge for edge. The next level's groups
// are drawn differently, so a border locked at one level is free at the
// next, and no seam survives more than one level.
//
// Choosing what to draw: each cluster stores the error of its own level and
// of the coarser level made from its group, each with a bounding sphere.
// Both are the same for every cluster of a group, and both only grow toward
// the root, so the test
//
//     own error on screen <= threshold < parent error on screen
//
// picks exactly one level along every path from a leaf to a root, and all
// clusters of one group agree. Each cluster can be tested on its own, in any
// order: a GPU thread per cluster, with no traversal.
#include "mesh.hpp"

#include <cmath>
#include <vector>

struct lod_cluster {
    std::vector<uint32_t> indices;  // Three per triangle, into lod_mesh::positions
    sphere bounds;                  // Around this cluster's vertices: for culling
    vec3 cone_axis;                 // Contains every face normal: see cone_cutoff
    // Sine of the widest angle between cone_axis and a face normal: every
    // triangle faces away from a viewer at unit direction d from the
    // bounds' center when dot(d, axis) >= cone_cutoff + radius / distance.
    // 1 turns the test off.
    float cone_cutoff = 1;

    sphere lod_bounds;  // Shared by every cluster made from the same group
    float lod_error = 0;
    sphere parent_bounds;              // The group this cluster was simplified in
    float parent_error = INFINITY;    // Infinite for a root: nothing coarser exists
    uint32_t level = 0;
    uint32_t group = UINT32_MAX;       // Number of the group it was simplified in, if any
};

struct lod_level_stats {
    size_t clusters = 0, triangles = 0, groups = 0, stuck_groups = 0;
    float max_error = 0;
};

struct lod_mesh {
    std::vector<vec3> positions, normals;
    std::vector<lod_cluster> clusters;  // Level 0 first, then each level in turn
    std::vector<lod_level_stats> levels;
};

// Builds the hierarchy over a welded mesh. Logs one line per level if
// verbose.
lod_mesh build_lod(const mesh& m, bool verbose);

// Cone and bounds for a cluster's triangles.
void cluster_bounds(const std::vector<vec3>& positions, lod_cluster& c);
