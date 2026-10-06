#pragma once
// edge collapse simplification with quadric error metrics (garland and heckbert
// 1997).
//
// each vertex carries a quadric: summed squared distances to its triangles'
// planes, area weighted, added on collapse. collapses keep an endpoint, so
// every lod uses the input's vertices and a model shares one vertex buffer.
#include "math.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

// a wedge made by a relaxed seam collapse (see wedges::relax).
struct new_wedge {
    vec2 uv;
    uint32_t vertex, chart;
    uint8_t material;
};
constexpr uint32_t new_wedge_bit = 0x80000000u;
// a vertex made by grow_to_area(): new_vertex_bit | k names new_positions[k].
constexpr uint32_t new_vertex_bit = 0x80000000u;

struct simplify_result {
    std::vector<uint32_t> indices;  // into the input's positions
    // a wedge per index, with wedges given; new_wedge_bit | k names new_wedges[k].
    std::vector<uint32_t> corners;
    std::vector<new_wedge> new_wedges;
    // vertex clustering: the vertex each input vertex snapped to.
    std::unordered_map<uint32_t, uint32_t> snapped;
    // grow_to_area(): new vertices, each a moved copy of new_from[k] (indices and new wedges
    // name them with new_vertex_bit).
    std::vector<vec3> new_positions;
    std::vector<uint32_t> new_from;
    // estimated distance from the input: root of the worst collapse's
    // area-weighted mean squared distance.
    float error = 0;
};

// texture wedges of a mesh being simplified: a wedge per input index, and each wedge's chart
// and vertex (lod_mesh).
struct wedges {
    const std::vector<uint32_t>& corners;
    const std::vector<uint32_t>& chart;
    const std::vector<uint32_t>& vertex;
    const std::vector<vec2>& uv;
    const std::vector<uint8_t>& material;
    // seams may move: a corner whose chart has no wedge at the vertex it moves onto gets a new
    // one there, keeping its old coordinates. positions stay shared, so no cracks; the texture
    // shifts across the seam by at most the collapse's distance, which the error measures.
    bool relax = false;
};

// collapses edges, cheapest first, down to target_triangles or until none is
// allowed. with wedges, a corner moving onto a vertex takes that vertex's wedge in
// the corner's chart, and a collapse is refused where the vertex has none: seam
// vertices only slide along their seams, and texture coordinates stay exact. locked[v] never moves (others may move onto it), which holds group
// outlines fixed. refuses flips and link-condition violations. border vertices
// only slide along the border, where planes through each border edge price
// moving the outline.
//
// with poses (each a full set of positions, the mesh posed: a skinned mesh's animations), each
// vertex also carries a quadric per pose, of its triangles posed, and a collapse costs what
// it costs in every pose (mohr and gleicher 2003): flat in the rest pose isn't flat when bent.
simplify_result simplify(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices,
                         const std::vector<uint8_t>& locked, size_t target_triangles, const wedges* w = nullptr,
                         const std::vector<std::vector<vec3>>* poses = nullptr);

// vertex clustering (rossignac and borrel): each vertex snaps to one vertex of
// its grid cell, the grid coarsened until at most target_triangles remain;
// collapsed triangles drop. ignores topology, so it gets anywhere edge
// collapses cannot; used only for the last group, which nothing borders. stays
// crack-free: duplicates are kept (edge use counts keep their parity) and a
// cell with a border vertex keeps one. error is 0: measure it.
// with wedges, a corner takes its cell's vertex's wedge in its chart (any wedge of that vertex
// where it has none: at the coarsest level a seam may smear).
//
// with locked (indexed by vertex), locked vertices never move: other vertices in their cell
// snap onto one of them, so a group's outline stays and its neighbours still meet it. this is
// what gets islands (needles, leaves, blades: pieces no collapse may remove) to merge.
simplify_result cluster_vertices(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices,
                                 size_t target_triangles, const wedges* w = nullptr,
                                 const std::vector<uint8_t>* locked = nullptr);

// after vertex clustering: thin pieces (needles, blades) thinner than a cell collapse and their
// area goes with them, so foliage thins at a distance. each piece of the result (triangles
// joined by vertices) takes the area of the input that snapped into it, or, where an input
// triangle vanished, the nearest piece takes it, and grows about its centre to match, at most
// max_scale across. locked vertices stay, so the outline does and cuts stay whole; moved
// vertices are new (new_positions), the originals untouched for the levels below. the error
// must be measured after.
void grow_to_area(const std::vector<vec3>& positions, const std::vector<uint32_t>& input, simplify_result& r,
                  const std::vector<uint8_t>& locked, const wedges* w, float max_scale);
