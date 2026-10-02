#version 460
// depth and triangle into the visibility buffer in one 64-bit atomic max:
// reversed depth, nearest wins. the depth attachment rejects most hidden
// fragments first.
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#include "common.glsl"

layout(early_fragment_tests) in;
layout(location = 0) perprimitiveEXT flat in uint id;

void main() {
    const uint px = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * frame.width;
    atomicMax(vis[px], (uint64_t(floatBitsToUint(gl_FragCoord.z)) << 32) | uint64_t(id));
}
