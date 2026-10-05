#pragma once
// gltf 2.0 reading (.gltf with its .bin, or .glb): triangles, texture coordinates, and for a
// skinned mesh its joints, weights, skeleton and animations. no new dependencies: a small
// json parser is inside.
#include "mesh.hpp"
#include "skeleton.hpp"

#include <string>

// every triangle primitive of the scene's meshes (only `node`'s, if named), placed by their
// nodes. a skinned mesh is read in its bind pose (gltf ignores its node's transform) and `skin`
// filled; only one skin is supported. if any material has a base colour texture the mesh is
// textured: each material its texture (texture transforms baked into the coordinates, repeat
// from its sampler) or its base colour, and whether it is double sided.
mesh load_gltf(const std::string& path, skeleton* skin, const std::string& node = "");
