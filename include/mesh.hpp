#pragma once
// indexed triangle mesh; ply and obj reading.
#include "math.hpp"

#include <string>
#include <vector>

struct mesh {
    std::vector<vec3> positions;
    std::vector<uint32_t> indices;  // three per triangle

    size_t triangle_count() const { return indices.size() / 3; }
};

// reads ply (binary, either byte order, or ascii) or obj, by extension.
// polygons become fans. throws on anything else.
mesh load_mesh(const std::string& path);

// merges vertices at one position, drops triangles using a vertex twice and
// repeated triangles. the simplifier needs one vertex per point.
void weld(mesh& m);

// stands the mesh on y = 0, centred on the y axis, largest side 1. up_z turns a
// z-up scan to y-up first. rewinds triangles to run counterclockwise from
// outside.
void normalize_placement(mesh& m, bool up_z);

// area-weighted vertex normals.
std::vector<vec3> vertex_normals(const mesh& m);
