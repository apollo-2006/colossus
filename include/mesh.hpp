#pragma once
// indexed triangle mesh; ply and obj reading.
#include "math.hpp"
#include "skeleton.hpp"

#include <string>
#include <vector>

// a material: its texture (a file, or where there is none a flat colour), whether its texture
// coordinates repeat (wrap) or clamp, and whether its triangles are seen from both sides; and
// optionally a normal map (tangent space, opengl's +y up) and a roughness map (its green
// channel, as gltf's metallicRoughnessTexture has it), sharing the texture's coordinates.
struct material {
    std::string texture;       // empty: the colour
    std::string normal_map, roughness_map;  // empty: none
    float color[3] = {0.8f, 0.8f, 0.8f};
    bool repeat = false;
    bool double_sided = false;
};

// a textured mesh keeps texture coordinates per wedge: a vertex as one texture chart sees it
// (a vertex on a uv seam has a wedge per chart). each triangle corner names its wedge, and each
// wedge its material (a triangle's corners share one). an untextured mesh has none.
struct mesh {
    std::vector<vec3> positions;
    std::vector<uint32_t> indices;  // three per triangle
    std::vector<uint32_t> corners;  // a wedge per index, or empty
    std::vector<vec2> wedge_uvs;    // texture coordinates, v down
    std::vector<uint32_t> wedge_vertex;
    std::vector<uint8_t> wedge_material;  // into materials
    std::vector<material> materials;
    // skinned: four joints and weights (summing to 255) per vertex, or empty.
    std::vector<uint8_t> skin_joints, skin_weights;

    size_t triangle_count() const { return indices.size() / 3; }
    bool textured() const { return !corners.empty(); }
    bool skinned() const { return !skin_weights.empty(); }
};

// reads ply (binary, either byte order, or ascii), obj or gltf (gltf.hpp), by extension.
// polygons become fans. throws on anything else. a skinned gltf's skeleton goes to `skin`.
mesh load_mesh(const std::string& path, skeleton* skin = nullptr, const std::string& node = "");

// merges vertices at one position, drops triangles using a vertex twice and
// repeated triangles. the simplifier needs one vertex per point.
void weld(mesh& m);

// stands the mesh on y = 0, centred on the y axis, largest side 1. up_z turns a
// z-up scan to y-up first. rewinds triangles to run counterclockwise from
// outside. returns the transform applied.
// keep_scale: stood and centred, but in its own units (a scene's models keep their sizes).
placement normalize_placement(mesh& m, bool up_z, bool keep_scale = false);

// loop subdivision (loop 1987), once: each triangle into four, new vertices from the edge
// rule, old ones smoothed; borders and edges of three or more triangles stay creases.
// texture coordinates and skin weights are interpolated linearly (weights keep the four
// largest).
void subdivide(mesh& m);

// area-weighted vertex normals.
std::vector<vec3> vertex_normals(const mesh& m);
