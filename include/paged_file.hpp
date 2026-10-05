#pragma once
// the hierarchy as shipped and streamed (.cgeo): clusters' bounds and errors
// always resident, geometry in pages, one per group, loaded on demand.
//
// a page holds its clusters' vertices then triangles, bit-packed, each cluster's
// two runs starting on a word. shared vertices are stored per cluster: more
// space, but a cluster reads only its page.
//
// positions snap to a grid over the model (step a 16382nd of the largest
// cluster's side), stored as offsets from the cluster's corner in just the bits
// its extent needs per axis (bx, by, bz, up to 14). normals are octahedral,
// 11 + 11 bits. a vertex is x (bx) | y (by) | z (bz) | u (11) | v (11), lowest
// bits first; a triangle three indices of ib bits, ib enough for the vertex
// count. the widths ride in the cluster's level word (cluster_level()):
//
//   level (8) | bx (4) << 8 | by (4) << 12 | bz (4) << 16 | ib (3) << 20 | textured (1) << 23
//
// a textured cluster's texture coordinates follow its vertex run (and the word after it): a
// word u0 | v0 << 16, the cluster's corner on a 65535-step grid over [0, 1], a word bu | bv << 5,
// then per vertex its offsets from the corner in bu and bv bits.
//
// shared vertices snap to the same grid point in each cluster: no cracks.
//
// clusters are stored packed (packed_cluster, 48 bytes): what a group's clusters
// share, their bounds and errors, is stored once per page (page_bounds).
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

// a cluster as stored and uploaded: gpu_cluster less what its pages share. its lod bounds and
// error are its creator page's page_bounds (a leaf's: its own sphere, error 0), its parent's
// its own page's. the rest packs:
//
//   cone     axis octahedral u (11) | v (11) << 11 | cutoff (10) << 22: the cutoff rounded up
//            and raised by how far the axis rounded, so it never culls more than the exact
//            cone. cutoff 1 (1023): no cone.
//   offsets  vertex_offset | triangle_offset << 16, words into the page
//   level    the level word above | vertex_count << 24
//   origin   24 bits each, origin[0] | triangle_count << 24
struct packed_cluster {
    float center[3];  // culling bounds
    float radius;
    uint32_t cone, offsets, level, group, creator;
    uint32_t origin[3];
};
static_assert(sizeof(packed_cluster) == 48);

// what a page's clusters share: their parent bounds and error, which are also the lod bounds
// and error of the clusters made from them. page 0, the roots', has an infinite error.
struct page_bounds {
    float center[3];
    float radius;
    float error;
};
static_assert(sizeof(page_bounds) == 20);

struct paged_geometry {
    sphere bounds, lod_bounds;
    vec3 grid_min;      // grid point (0, 0, 0)
    float grid_step = 0;
    // by parent error. `group` is the cluster's page, `creator` the page of the
    // finer clusters it stands for (no_page for a leaf); vertex_offset and
    // triangle_offset are words into the page. unpacked from packed and
    // shared_bounds: exactly what the file holds.
    std::vector<gpu_cluster> clusters;
    std::vector<packed_cluster> packed;
    std::vector<page_bounds> shared_bounds;  // per page
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

packed_cluster pack_cluster(const gpu_cluster& c);
gpu_cluster unpack_cluster(const packed_cluster& p, const std::vector<page_bounds>& shared_bounds);

// a paged cluster's level, its widths masked off.
inline uint32_t cluster_level(const gpu_cluster& c) { return c.level & 0xff; }
// vertex k of a cluster, from its page's words.
vec3 decode_position(const paged_geometry& g, const gpu_cluster& c, const uint32_t* page, uint32_t k);
vec3 decode_normal(const gpu_cluster& c, const uint32_t* page, uint32_t k);
// texture coordinates of vertex k of a textured cluster.
vec2 decode_uv(const gpu_cluster& c, const uint32_t* page, uint32_t k);
inline bool cluster_textured(const gpu_cluster& c) { return (c.level >> 23) & 1; }
// triangle t of a cluster (a | b << 8 | c << 16) from its page.
uint32_t decode_triangle(const gpu_cluster& c, const uint32_t* page, uint32_t t);
