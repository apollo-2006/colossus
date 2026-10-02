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
// Each piece of the mesh is cut in two, again and again, along its
// triangle graph (triangles joined across edges): every triangle goes to
// whichever of two far-apart seed triangles it is fewer steps from, the cut
// falling at a whole number of clusters, and a few passes of swaps then
// shorten the cut. So every cluster but one per piece is full, and both
// halves of every cut are compact. Large inputs are first cut in half
// along their longest axis until each piece is small, and the pieces are
// clustered in parallel.
//
// This replaced greedy growth (each cluster taking the neighbouring
// triangle that added the fewest vertices), which filled clusters to 108
// triangles on average on the XYZ RGB dragon and left 8% of them as
// pockets of a few triangles walled in by full ones. Bisection fills all
// of them, 20% fewer clusters, and the hierarchy above ends in one root.
std::vector<std::vector<uint32_t>> clusterize(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices);
