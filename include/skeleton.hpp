#pragma once
// a skinned model's skeleton and animations (.cskn beside the .cgeo), from gltf.
//
// joint matrices: a joint's node's world transform (its ancestors' local transforms, each
// translation * rotation * scale) times its inverse bind matrix. the builder moved the model
// into place (normalize_placement()), so the viewer conjugates them by that placement.
//
// pages carry each vertex's joints and weights (paged_file.hpp); per page the file also keeps
// which joints its group's vertices use (page_skin), which bound how far skinning moves a
// cluster. errors already hold the animations (build_lod()).
#include "math.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct skeleton_node {
    int32_t parent = -1;
    float translation[3] = {0, 0, 0};
    float rotation[4] = {0, 0, 0, 1};  // quaternion x y z w
    float scale[3] = {1, 1, 1};
};

struct animation_channel {
    uint32_t node = 0;
    uint32_t path = 0;    // 0 translation, 1 rotation, 2 scale
    uint32_t step = 0;    // 1: step interpolation, else linear (cubic splines keep their values)
    std::vector<float> times, values;  // values: 3 or 4 per key
};

struct animation {
    std::string name;
    float duration = 0;
    std::vector<animation_channel> channels;
};

// per page: the joints its group's vertices use (bit j, joints under 64), only growing toward
// the roots (paged_file.cpp).
struct page_skin {
    uint64_t joints = 0;
};
static_assert(sizeof(page_skin) == 8);

struct placement {  // p -> (turned(p) - base) * scale; turned swaps a z-up model to y-up
    uint32_t up_z = 0;
    vec3 base;
    float scale = 1;
};

struct skeleton {
    std::vector<skeleton_node> nodes;
    std::vector<uint32_t> joints;       // nodes
    std::vector<float> inverse_bind;    // 16 per joint, column major
    std::vector<animation> animations;
    placement place;
    std::vector<page_skin> pages;
    // the joint carrying the most weight (a hip): lod spheres follow it, grown by how far the
    // other joints move points relative to it, so moving the whole model costs nothing.
    uint32_t anchor = 0;
};

void save_skeleton(const skeleton& s, const std::string& path);
skeleton load_skeleton(const std::string& path);

// joint matrices (3x4, rows) of an animation at time t (looped), in the placed model's space.
// animation -1: the rest pose.
void pose_joints(const skeleton& s, int animation_index, float t, std::vector<float>& rows);
