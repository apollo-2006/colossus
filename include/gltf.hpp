#pragma once
// gltf 2.0 reading (.gltf with its .bin, or .glb): triangles, texture coordinates, and for a
// skinned mesh its joints, weights, skeleton and animations. no new dependencies: a small
// json parser is inside.
#include "mesh.hpp"
#include "skeleton.hpp"

#include <string>

// every triangle primitive of the scene's meshes, placed by their nodes. a skinned mesh is
// read in its bind pose (gltf ignores its node's transform) and `skin` filled; only one skin
// is supported. the base colour texture of the first material that has one, if any.
mesh load_gltf(const std::string& path, skeleton* skin);
