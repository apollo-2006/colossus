#pragma once
// indexed triangle mesh; ply and obj reading.
#include "math.hpp"
#include "skeleton.hpp"

#include <string>
#include <vector>

// a textured mesh keeps texture coordinates per wedge: a vertex as one texture chart sees it
// (a vertex on a uv seam has a wedge per chart). each triangle corner names its wedge. an
// untextured mesh has none.
struct mesh {
    std::vector<vec3> positions;
    std::vector<uint32_t> indices;  // three per triangle
    std::vector<uint32_t> corners;  // a wedge per index, or empty
    std::vector<vec2> wedge_uvs;    // texture coordinates, v down
    std::vector<uint32_t> wedge_vertex;
    std::string texture;            // the texture's file, if any (an obj's map_Kd)
    // skinned: four joints and weights (summing to 255) per vertex, or empty.
    std::vector<uint8_t> skin_joints, skin_weights;

    size_t triangle_count() const { return indices.size() / 3; }
    bool textured() const { return !corners.empty(); }
    bool skinned() const { return !skin_weights.empty(); }
};

// reads ply (binary, either byte order, or ascii), obj or gltf (gltf.hpp), by extension.
// polygons become fans. throws on anything else. a skinned gltf's skeleton goes to `skin`.
mesh load_mesh(const std::string& path, skeleton* skin = nullptr);

// merges vertices at one position, drops triangles using a vertex twice and
// repeated triangles. the simplifier needs one vertex per point.
void weld(mesh& m);

// stands the mesh on y = 0, centred on the y axis, largest side 1. up_z turns a
// z-up scan to y-up first. rewinds triangles to run counterclockwise from
// outside. returns the transform applied.
placement normalize_placement(mesh& m, bool up_z);

// loop subdivision (loop 1987), once: each triangle into four, new vertices from the edge
// rule, old ones smoothed; borders and edges of three or more triangles stay creases.
// texture coordinates and skin weights are interpolated linearly (weights keep the four
// largest).
void subdivide(mesh& m);

// area-weighted vertex normals.
std::vector<vec3> vertex_normals(const mesh& m);
