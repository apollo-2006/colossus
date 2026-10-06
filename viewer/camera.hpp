#pragma once
// the camera and its frustum planes.
#include "math.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

// a camera path for recordings (--path): keyframes `t x y z yaw pitch [threshold]`, seconds
// from the first recorded frame, `#` comments. positions and angles follow a catmull-rom
// curve through the keys, the threshold moves geometrically between them.
struct camera_key {
    float t;
    vec3 eye;
    float yaw, pitch, threshold;
};

std::vector<camera_key> read_camera_path(const std::string& path, float default_threshold) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<camera_key> keys;
    std::string line;
    while (std::getline(in, line)) {
        if (const size_t hash = line.find('#'); hash != std::string::npos) line.resize(hash);
        std::istringstream words(line);
        camera_key k{};
        if (!(words >> k.t)) continue;
        if (!(words >> k.eye.x >> k.eye.y >> k.eye.z >> k.yaw >> k.pitch)) throw std::runtime_error(path + ": bad key: " + line);
        if (!(words >> k.threshold)) k.threshold = default_threshold;
        if (!keys.empty() && k.t <= keys.back().t) throw std::runtime_error(path + ": keys must go forward in time");
        keys.push_back(k);
    }
    if (keys.empty()) throw std::runtime_error(path + ": no keys");
    return keys;
}

// the path at time t (held at its ends): the camera and the threshold.
void sample_camera_path(const std::vector<camera_key>& keys, float t, camera& cam, float& threshold) {
    size_t i = 0;
    while (i + 1 < keys.size() && keys[i + 1].t <= t) ++i;
    if (i + 1 >= keys.size() || t <= keys[0].t) {
        const camera_key& k = t <= keys[0].t ? keys[0] : keys.back();
        cam.eye = k.eye; cam.yaw = k.yaw; cam.pitch = k.pitch; threshold = k.threshold;
        return;
    }
    const camera_key& a = keys[i == 0 ? 0 : i - 1];
    const camera_key& b = keys[i];
    const camera_key& c = keys[i + 1];
    const camera_key& d = keys[i + 2 < keys.size() ? i + 2 : i + 1];
    const float u = (t - b.t) / (c.t - b.t);
    auto spline = [u](float p0, float p1, float p2, float p3) {
        return 0.5f * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (3 * p1 - p0 - 3 * p2 + p3) * u * u * u);
    };
    cam.eye = {spline(a.eye.x, b.eye.x, c.eye.x, d.eye.x), spline(a.eye.y, b.eye.y, c.eye.y, d.eye.y),
               spline(a.eye.z, b.eye.z, c.eye.z, d.eye.z)};
    cam.yaw = spline(a.yaw, b.yaw, c.yaw, d.yaw);
    cam.pitch = spline(a.pitch, b.pitch, c.pitch, d.pitch);
    threshold = b.threshold * std::pow(c.threshold / b.threshold, u * u * (3 - 2 * u));
}

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
