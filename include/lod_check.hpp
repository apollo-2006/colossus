#pragma once
// Checks that the hierarchy joins up: for a range of error thresholds, the
// cut it selects (each cluster with lod_error <= t < parent_error) must
// cover the model once and leave no cracks.
//
// Every level shares the original vertices, so a crack is easy to see
// exactly: an edge used by one triangle of the cut whose ends are not both
// on an open border of the original mesh. Coverage is checked by area,
// which simplification only changes a little.
#include "geometry_file.hpp"

#include <string>
#include <vector>

struct cut_report {
    float threshold = 0;
    size_t clusters = 0, triangles = 0;
    size_t cracked_edges = 0;  // Must be 0
    double area_ratio = 0;     // The cut's area over the original's: near 1
};

// Thresholds in model units, from 0 (the leaves) to past the root's error.
std::vector<cut_report> check_cuts(const geometry& g, int steps);
