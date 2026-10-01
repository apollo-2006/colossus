#pragma once
// Mesh simplification by edge collapse with quadric error metrics (Garland
// and Heckbert 1997), for the LOD builder.
//
// Each vertex carries a quadric: the sum of squared distances to the planes
// of the triangles around it, area weighted, which collapses add together,
// so a vertex remembers every plane the surface it stands for used to touch.
// Collapses are endpoint collapses: one end of an edge moves onto the other,
// so the simplified mesh only ever uses vertices of the input. Every LOD of
// a model therefore shares one vertex buffer.
#include "math.hpp"

#include <cstdint>
#include <vector>

struct simplify_result {
    std::vector<uint32_t> indices;  // Into the same positions as the input
    // Estimated distance between the result and the input: the square root
    // of the largest area-weighted mean squared distance of any collapse.
    float error = 0;
};

// Collapses edges, cheapest first, until at most target_triangles remain or
// no collapse is allowed. A vertex with locked[v] set never moves (other
// vertices can still move onto it), which keeps the outline of a cluster
// group fixed so it meets its neighbours at every LOD. Collapses that would
// flip a triangle, or join two sheets of surface at a vertex (the link
// condition), are refused. A vertex on an open border of the mesh may only
// collapse along the border, and planes through each border edge,
// perpendicular to its triangle, make that cost what it moves the outline.
simplify_result simplify(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices,
                         const std::vector<uint8_t>& locked, size_t target_triangles);
