#pragma once
// the hierarchy as shipped and streamed (.cgeo): clusters' bounds and errors
// always resident, geometry in pages, one per group, loaded on demand.
//
// a page holds its clusters' vertices then triangles (three bytes each), packed
// end to end, each cluster padded to a word. shared vertices are stored per
// cluster: more space, but a cluster reads only its page.
//
// a vertex is two words: position snapped to a grid over the model (step a
// 16382nd of the largest cluster's side) as three 14-bit offsets from the
// cluster's corner, and an octahedral normal, 11 + 11 bits:
//
//   word 0: x (14) | y (14) << 14 | low 4 bits of z << 28
//   word 1: high 10 bits of z | u (11) << 10 | v (11) << 21
//
// shared vertices snap to the same grid point in each cluster: no cracks.
//
// pages are the unit of streaming. page 0 holds the roots, always resident. a
// page's dependencies hold the coarser copy of its surface, and it is resident
// only while they are, so a cluster can always stand in for missing finer ones:
// see viewer/streamer.hpp.
#include "geometry_file.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct page_info {
    uint64_t offset;      // bytes into the page data
    uint32_t size;        // bytes, a multiple of 16
    uint32_t dep_first;   // into paged_geometry::deps
    uint32_t dep_count;
    uint32_t cluster_count;
};
static_assert(sizeof(page_info) == 24);

constexpr uint32_t no_page = UINT32_MAX;

struct paged_geometry {
    sphere bounds, lod_bounds;
    vec3 grid_min;      // grid point (0, 0, 0)
    float grid_step = 0;
    // by parent error. `group` is the cluster's page, `creator` the page of the
    // finer clusters it stands for (no_page for a leaf); vertex_offset and
    // triangle_offset are words into the page.
    std::vector<gpu_cluster> clusters;
    std::vector<page_info> pages;
    std::vector<uint32_t> deps;
    std::vector<lod_level_stats> levels;
    uint64_t data_offset = 0;     // where page data starts in the file
    uint64_t data_size = 0;
    std::vector<uint32_t> data;   // page data, if loaded

    size_t leaf_triangles() const;
};

// packs geometry into pages.
paged_geometry page(const geometry& g);

void save_paged(const paged_geometry& p, const std::string& path);
// reads a .cgeo; page data only if with_data.
paged_geometry load_paged(const std::string& path, bool with_data);

// a cluster vertex from its two words.
vec3 decode_position(const paged_geometry& g, const gpu_cluster& c, const uint32_t* vertex);
// triangle t of a cluster (a | b << 8 | c << 16) from its page.
uint32_t decode_triangle(const gpu_cluster& c, const uint32_t* page, uint32_t t);
vec3 decode_normal(const uint32_t* vertex);
