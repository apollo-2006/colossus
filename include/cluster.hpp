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
// the cluster that adds the fewest new vertices, stays nearest the
// cluster's center, and has the fewest unused neighbours (so notches that
// would become islands are taken first). Taking triangles that close a gap
// first keeps the outline short, which matters twice over: fewer shared
// vertices per triangle, and fewer locked edges when the LOD builder
// simplifies a group of clusters. Each new cluster starts on the last
// one's frontier, at its most hemmed-in triangle. Large inputs are first
// cut in half along their longest axis until each piece is small, and the
// pieces are clustered in parallel.
//
// On the XYZ RGB dragon this fills clusters to 108 triangles on average
// (Morton-order seeds alone: 90). About 8% are still pockets of under 16
// triangles walled in by full clusters; swapping triangles between
// neighbours, or a real graph partitioner, would take those too.
std::vector<std::vector<uint32_t>> clusterize(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices);
