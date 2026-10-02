#pragma once
// distance between two meshes: the largest from a sample on either (vertices,
// edge midpoints, triangle centres) to the other. a sampled two-sided hausdorff
// distance; the builder's level error, since the quadric error is a mean and
// understates the worst spot.
#include "math.hpp"

#include <cstdint>
#include <vector>

float mesh_deviation(const std::vector<vec3>& positions, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b);

// distance from p to triangle abc (ericson, real-time collision detection,
// 5.1.5).
float point_triangle_distance(vec3 p, vec3 a, vec3 b, vec3 c);
