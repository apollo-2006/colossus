#pragma once
// An indexed triangle mesh, and reading one from PLY or OBJ.
#include "math.hpp"

#include <string>
#include <vector>

struct mesh {
    std::vector<vec3> positions;
    std::vector<uint32_t> indices;  // Three per triangle

    size_t triangle_count() const { return indices.size() / 3; }
};

// Reads binary (either byte order) or ASCII PLY, or Wavefront OBJ, by the
// file's extension. Polygons are split into fans. Throws on anything else.
mesh load_mesh(const std::string& path);

// Merges vertices at the same position, then drops triangles that use a
// vertex twice and second copies of a triangle. Scans are already indexed,
// but OBJ files and some PLY writers repeat vertices per face, and the
// simplifier needs one vertex per point to see which triangles meet.
void weld(mesh& m);

// Moves and scales the mesh to stand on y = 0, centered on the y axis, with
// its largest side 1 long. up_z first turns a z-up scan to y-up. Triangles
// are rewound, if need be, to run counterclockwise seen from outside.
void normalize_placement(mesh& m, bool up_z);

// Area-weighted vertex normals.
std::vector<vec3> vertex_normals(const mesh& m);
