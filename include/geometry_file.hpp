#pragma once
// The built hierarchy as the GPU reads it, and its file (.ngeo), which is
// the same bytes: the viewer reads a file straight into buffers.
//
// Each cluster lists the vertices it uses (indices into the shared vertex
// arrays) and its triangles as three bytes indexing that list, packed into
// one uint32 each, the layout a mesh shader workgroup reads.
#include "cluster.hpp"
#include "dag.hpp"

#include <string>
#include <vector>

// Laid out as the shaders read it (scalar layout): 96 bytes.
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
    uint32_t group;  // Unique over the whole hierarchy, for debug colors
};
static_assert(sizeof(gpu_cluster) == 96, "gpu_cluster must match the shaders");

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
void save_geometry(const geometry& g, const std::string& path);
geometry load_geometry(const std::string& path);
