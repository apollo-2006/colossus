#version 460
// Writes depth and triangle into the visibility buffer with one 64-bit
// atomic max: depth is reversed, so the nearest wins. The depth attachment
// still rejects most hidden fragments before they get here.
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#include "common.glsl"

layout(early_fragment_tests) in;
layout(location = 0) perprimitiveEXT flat in uint id;

void main() {
    const uint px = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * frame.width;
    atomicMax(vis[px], (uint64_t(floatBitsToUint(gl_FragCoord.z)) << 32) | uint64_t(id));
}
