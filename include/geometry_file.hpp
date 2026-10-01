#pragma once
// The built hierarchy, with the clusters as the GPU reads them and every
// level indexing one shared vertex array: the form it is checked and
// trimmed in, before page() packs it for shipping (paged_file.hpp).
//
// Each cluster lists the vertices it uses (indices into the shared vertex
// arrays) and its triangles as three bytes indexing that list, packed into
// one uint32 each, the layout a mesh shader workgroup reads.
#include "cluster.hpp"
#include "dag.hpp"

#include <string>
#include <vector>

// Laid out as the shaders read it: 112 bytes, a multiple of 16 for WGSL.
struct gpu_cluster {
    float center[3];  // Culling bounds
    float radius;
    float cone_axis[3];
    float cone_cutoff;
    float lod_center[3];
    float lod_radius;
    float parent_center[3];
    float parent_radius;
    float lod_error;
    float parent_error;  // FLT_MAX for a root
    uint32_t vertex_offset;    // Into cluster_vertices
    uint32_t triangle_offset;  // Into cluster_triangles
    uint32_t vertex_count;
    uint32_t triangle_count;
    uint32_t level;
    uint32_t group;    // The group it was simplified in (UINT32_MAX for a root)
    uint32_t creator;  // The group whose simplification made it (UINT32_MAX for a leaf)
    uint32_t pad0, pad1, pad2;
};
static_assert(sizeof(gpu_cluster) == 112, "gpu_cluster must match the shaders");

struct geometry {
    std::vector<float> positions;  // xyz per vertex
    std::vector<float> normals;    // xyz per vertex
    std::vector<gpu_cluster> clusters;  // In order of parent error
    std::vector<uint32_t> cluster_vertices;
    std::vector<uint32_t> cluster_triangles;  // a | b << 8 | c << 16
    std::vector<lod_level_stats> levels;
    sphere bounds;      // Around the whole model
    sphere lod_bounds;  // Around bounds and every cluster's LOD and parent spheres

    size_t leaf_triangles() const;
};

geometry pack(const lod_mesh& lod);

// Drops the finest levels, keeping the finest cut with at most
// max_triangles triangles as the new leaves (their error becomes 0) and
// every cluster that can be drawn above it. Each kept leaf's parent is
// unchanged, so cuts stay crack-free. Unused vertices are dropped. For
// shipping a model somewhere memory is short, like a web page.
geometry trim(const geometry& g, size_t max_triangles);
