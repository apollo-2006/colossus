#pragma once
// checks the hierarchy joins up: for a range of thresholds, the cut (lod_error
// <= t < parent_error) must cover the model once with no cracks.
//
// levels share the original vertices, so a crack is exact: an edge used by one
// cut triangle whose ends are not both on an original border. coverage is
// checked by area.
#include "geometry_file.hpp"

#include <string>
#include <vector>

struct cut_report {
    float threshold = 0;
    size_t clusters = 0, triangles = 0;
    size_t cracked_edges = 0;  // must be 0
    double area_ratio = 0;     // cut area over original: near 1
};

// thresholds in model units, from 0 (leaves) past the root's error.
std::vector<cut_report> check_cuts(const geometry& g, int steps);
