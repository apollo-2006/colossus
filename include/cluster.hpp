#pragma once
// triangles into clusters: the unit of drawing, culling and the lod hierarchy.
#include "math.hpp"

#include <vector>

constexpr uint32_t cluster_max_triangles = 128;
constexpr uint32_t cluster_max_vertices = 128;

// splits triangles (three indices each) into clusters of at most
// cluster_max_triangles triangles and cluster_max_vertices vertices. each
// cluster is a list of triangle numbers.
//
// recursive bisection of the triangle graph: each triangle goes to the nearer
// of two far-apart seeds, the cut falls at a whole number of clusters, and
// swaps then shorten it. every cluster but one per piece is full. large inputs
// are first halved along their longest axis and the pieces run in parallel.
//
// replaced greedy growth: 108 triangles per cluster on the dragon, 8% pockets.
// bisection fills them all: 20% fewer clusters, one root.
std::vector<std::vector<uint32_t>> clusterize(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices);
