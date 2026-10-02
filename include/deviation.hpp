#pragma once
// How far apart two triangle meshes are: the largest distance from a
// sample point on either to the other surface, the samples being every
// vertex, edge midpoint and triangle center. A sampled two-sided Hausdorff
// distance: the LOD builder takes it as a level's error, since the quadric
// error the simplifier minimizes is an area-weighted mean and can
// understate the worst spot.
#include "math.hpp"

#include <cstdint>
#include <vector>

float mesh_deviation(const std::vector<vec3>& positions, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b);

// The distance from p to triangle abc (Ericson, Real-Time Collision
// Detection, 5.1.5).
float point_triangle_distance(vec3 p, vec3 a, vec3 b, vec3 c);
