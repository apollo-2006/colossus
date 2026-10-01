#pragma once
// Splitting triangles into clusters: the unit everything else works in. A
// mesh shader workgroup draws one cluster, the culling passes test one
// cluster's bounds, and the LOD hierarchy is built from groups of them.
#include "math.hpp"

#include <vector>

constexpr uint32_t cluster_max_triangles = 128;
constexpr uint32_t cluster_max_vertices = 128;

// Splits the triangles (three indices each into positions) into clusters of
// at most cluster_max_triangles triangles using at most cluster_max_vertices
// distinct vertices. Each cluster is a list of triangle numbers.
//
// Clusters grow greedily from a seed: the next triangle is the one touching
// the cluster that adds the fewest new vertices, ties going to the one
// nearest the cluster's center. Taking triangles that close a gap first
// keeps the outline short, which matters twice over: fewer shared vertices
// per triangle, and fewer locked edges when the LOD builder simplifies a
// group of clusters. Large inputs are first cut in half along their longest
// axis until each piece is small, and the pieces are clustered in parallel.
std::vector<std::vector<uint32_t>> clusterize(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices);
