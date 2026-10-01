#pragma once
// The hierarchy as it is shipped and streamed (.cgeo files): the clusters'
// bounds and errors, always resident, and their geometry in pages, one per
// group, loaded as needed.
//
// A page holds everything its clusters draw from: for each cluster, its
// vertices (position, and normal packed octahedrally into 16 + 16 bits:
// four words each) and then its triangles (three bytes in a word). The
// vertices a cluster shares with its neighbours are stored again, which
// costs space but leaves a cluster nothing to look up but its own page.
//
// A page is the unit of streaming. Page 0 holds the roots and is always
// resident. A page is the members of one group; the clusters its group was
// simplified into (the coarser copy of the same surface) are in other
// pages, its dependencies, and a page may only be resident while those
// are. Then a cluster can always be drawn in place of finer clusters that
// are missing, and the cut stays whole: see viewer/residency.hpp.
#include "geometry_file.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct page_info {
    uint64_t offset;      // Bytes into the page data
    uint32_t size;        // Bytes, a multiple of 16
    uint32_t dep_first;   // Into paged_geometry::deps
    uint32_t dep_count;
    uint32_t cluster_count;
};
static_assert(sizeof(page_info) == 24);

constexpr uint32_t no_page = UINT32_MAX;

struct paged_geometry {
    sphere bounds, lod_bounds;
    // In order of parent error, like geometry's. Here `group` is the page a
    // cluster lives in and `creator` the page of the finer clusters it
    // stands for (no_page for a leaf); vertex_offset and triangle_offset
    // are words into that page.
    std::vector<gpu_cluster> clusters;
    std::vector<page_info> pages;
    std::vector<uint32_t> deps;
    std::vector<lod_level_stats> levels;
    uint64_t data_offset = 0;     // Where the page data starts in the file
    uint64_t data_size = 0;
    std::vector<uint32_t> data;   // The page data, if loaded

    size_t leaf_triangles() const;
};

// Packs geometry into pages.
paged_geometry page(const geometry& g);

void save_paged(const paged_geometry& p, const std::string& path);
// Reads a .cgeo file; the page data only if with_data.
paged_geometry load_paged(const std::string& path, bool with_data);

// A vertex as stored in a page.
struct page_vertex {
    float x, y, z;
    uint32_t normal;  // Octahedral, two snorm16
};
uint32_t encode_normal(vec3 n);
vec3 decode_normal(uint32_t packed);
