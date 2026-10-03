#pragma once
// distance between two meshes: an upper bound on the two-sided hausdorff distance,
// the largest distance from any point of either to the other. the builder's level
// error, since the quadric error is a mean and understates the worst spot.
//
// over a triangle, the distance to one triangle of the other mesh is convex, so at most
// its largest at the corners, and the distance to the mesh at most that: the smallest
// such over nearby triangles bounds it, exactly where one triangle is nearest
// throughout. (and distance is 1-lipschitz: at most the largest at the corners plus the
// longest edge over sqrt 3.) triangles whose bound could still beat the worst distance
// found split in four, until each bound is within `slack` of it (or under `tolerance`).
// the result is always a bound: refining only tightens it.
#include "math.hpp"

#include <cstdint>
#include <vector>

float mesh_deviation(const std::vector<vec3>& positions, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                     float slack = 0.1f, float tolerance = 0);

// the largest distance from samples on either (vertices, edge midpoints, triangle
// centres) to the other: a lower bound on the same, for tests.
float sampled_deviation(const std::vector<vec3>& positions, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b);

// distance from p to triangle abc (ericson, real-time collision detection,
// 5.1.5).
float point_triangle_distance(vec3 p, vec3 a, vec3 b, vec3 c);
