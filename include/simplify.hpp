#pragma once
// edge collapse simplification with quadric error metrics (garland and heckbert
// 1997).
//
// each vertex carries a quadric: summed squared distances to its triangles'
// planes, area weighted, added on collapse. collapses keep an endpoint, so
// every lod uses the input's vertices and a model shares one vertex buffer.
#include "math.hpp"

#include <cstdint>
#include <vector>

struct simplify_result {
    std::vector<uint32_t> indices;  // into the input's positions
    // estimated distance from the input: root of the worst collapse's
    // area-weighted mean squared distance.
    float error = 0;
};

// collapses edges, cheapest first, down to target_triangles or until none is
// allowed. locked[v] never moves (others may move onto it), which holds group
// outlines fixed. refuses flips and link-condition violations. border vertices
// only slide along the border, where planes through each border edge price
// moving the outline.
simplify_result simplify(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices,
                         const std::vector<uint8_t>& locked, size_t target_triangles);

// vertex clustering (rossignac and borrel): each vertex snaps to one vertex of
// its grid cell, the grid coarsened until at most target_triangles remain;
// collapsed triangles drop. ignores topology, so it gets anywhere edge
// collapses cannot; used only for the last group, which nothing borders. stays
// crack-free: duplicates are kept (edge use counts keep their parity) and a
// cell with a border vertex keeps one. error is 0: measure it.
simplify_result cluster_vertices(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices,
                                 size_t target_triangles);
