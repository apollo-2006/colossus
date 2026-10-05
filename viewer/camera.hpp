#pragma once
// the camera and its frustum planes.
#include "math.hpp"

#include <array>
#include <cmath>

namespace viewer {

struct camera {
    vec3 eye{0, 0.6f, 2.5f};
    float yaw = 0, pitch = -0.15f;  // yaw 0 looks down -z
    float fov = 1.0f;               // vertical, radians
    float near_z = 0.01f;

    vec3 forward() const {
        return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw)};
    }
    mat4 view_proj(float aspect) const {
        return perspective_reverse_z(fov, aspect, near_z) * look_to(eye, forward(), {0, 1, 0});
    }
};

// frustum planes from a view-projection (gribb and hartmann): left, right, bottom, top, near,
// normalized. no far plane.
void frustum_planes(const mat4& m, float out[5][4]) {
    auto row = [&](int r) { return std::array<float, 4>{m.at(r, 0), m.at(r, 1), m.at(r, 2), m.at(r, 3)}; };
    const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    const std::array<float, 4> planes[5] = {
        {r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]},
        {r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]},
        {r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]},
        {r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]},
        {r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]},  // reversed depth: z <= w
    };
    for (int k = 0; k < 5; ++k) {
        const float l = std::sqrt(planes[k][0] * planes[k][0] + planes[k][1] * planes[k][1] + planes[k][2] * planes[k][2]);
        for (int c = 0; c < 4; ++c) out[k][c] = planes[k][c] / l;
    }
}

}  // namespace viewer
