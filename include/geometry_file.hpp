#pragma once
// the built hierarchy, clusters as the gpu reads them, all levels indexing one
// vertex array: checked and trimmed here before page() packs it
// (paged_file.hpp).
//
// each cluster lists its vertices (into the shared arrays) and its triangles as
// three bytes into that list, one uint32 each.
#include "cluster.hpp"
#include "dag.hpp"

#include <string>
#include <vector>

// as the shaders read it: 112 bytes, a multiple of 16 for wgsl.
struct gpu_cluster {
    float center[3];  // culling bounds
    float radius;
    float cone_axis[3];
    float cone_cutoff;
    float lod_center[3];
    float lod_radius;
    float parent_center[3];
    float parent_radius;
    float lod_error;
    float parent_error;  // FLT_MAX for a root
    uint32_t vertex_offset;    // into cluster_vertices
    uint32_t triangle_offset;  // into cluster_triangles
    uint32_t vertex_count;
    uint32_t triangle_count;
    uint32_t level;
    uint32_t group;    // group it was simplified in (UINT32_MAX for a root)
    uint32_t creator;  // group whose simplification made it (UINT32_MAX for a leaf)
    uint32_t origin[3];  // paged: the cluster's corner on the model's grid (paged_file.hpp)
};
static_assert(sizeof(gpu_cluster) == 112, "gpu_cluster must match the shaders");

struct geometry {
    std::vector<float> positions;  // xyz per vertex
    std::vector<float> normals;    // xyz per vertex
    std::vector<gpu_cluster> clusters;  // by parent error
    // a cluster's vertices are wedges: a vertex and its texture coordinates in one chart.
    // untextured, both are empty and a wedge is its vertex.
    std::vector<uint32_t> wedge_vertex;
    std::vector<vec2> wedge_uvs;
    std::vector<uint8_t> wedge_material;
    std::vector<material> materials;
    // texture coordinates are stored as fractions of their range (they may repeat past 0 to
    // 1): uv = uv_min + stored * uv_extent.
    vec2 uv_min;
    float uv_extent = 1;
    std::vector<uint8_t> skin_joints, skin_weights;  // per vertex, four each, if skinned
    std::vector<uint32_t> cluster_vertices;  // wedges
    std::vector<uint32_t> cluster_triangles;  // a | b << 8 | c << 16
    std::vector<lod_level_stats> levels;
    sphere bounds;      // around the model
    sphere lod_bounds;  // around bounds and every lod and parent sphere

    size_t leaf_triangles() const;
    bool textured() const { return !wedge_uvs.empty(); }
    bool skinned() const { return !skin_weights.empty(); }
    uint32_t vertex_of(uint32_t wedge) const { return wedge_vertex.empty() ? wedge : wedge_vertex[wedge]; }
};

geometry pack(const lod_mesh& lod);

// drops the finest levels: the finest cut within max_triangles becomes the
// leaves (error 0), with everything drawable above it. parents unchanged, so
// cuts stay crack-free. unused vertices dropped. for memory-short targets like
// the web.
geometry trim(const geometry& g, size_t max_triangles);
