#pragma once
// Small vector and matrix types for the builder and the viewer. Matrices are
// column major, as GLSL reads a mat4.
#include <algorithm>
#include <cmath>
#include <cstdint>

struct vec3 {
    float x = 0, y = 0, z = 0;
    vec3() = default;
    constexpr vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    float operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
    float& operator[](int i) { return i == 0 ? x : i == 1 ? y : z; }
};

inline vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline vec3 operator-(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline vec3 operator-(vec3 a) { return {-a.x, -a.y, -a.z}; }
inline vec3 operator*(vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline vec3 operator*(float s, vec3 a) { return a * s; }
inline vec3& operator+=(vec3& a, vec3 b) { return a = a + b; }
inline float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline vec3 cross(vec3 a, vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float length(vec3 a) { return std::sqrt(dot(a, a)); }
inline vec3 normalize(vec3 a) {
    const float l = length(a);
    return l > 0 ? a * (1 / l) : vec3(0, 0, 0);
}
inline vec3 min(vec3 a, vec3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
inline vec3 max(vec3 a, vec3 b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }

struct sphere {
    vec3 center;
    float radius = 0;
};

// The smallest sphere around a and b.
inline sphere merge(const sphere& a, const sphere& b) {
    const vec3 d = b.center - a.center;
    const float dist = length(d);
    if (dist + b.radius <= a.radius) return a;
    if (dist + a.radius <= b.radius) return b;
    const float r = 0.5f * (dist + a.radius + b.radius);
    return {a.center + d * ((r - a.radius) / dist), r};
}

// A sphere around points, by Ritter's method: a start from two far apart
// points, grown to take in any point left outside. Within a few percent of
// the smallest.
template <class Get>
sphere bounding_sphere(size_t n, Get&& point) {
    if (n == 0) return {};
    auto farthest = [&](vec3 from) {
        size_t best = 0;
        float best_d = -1;
        for (size_t i = 0; i < n; ++i) {
            const vec3 d = point(i) - from;
            if (dot(d, d) > best_d) { best_d = dot(d, d); best = i; }
        }
        return point(best);
    };
    const vec3 a = farthest(point(0));
    const vec3 b = farthest(a);
    sphere s{(a + b) * 0.5f, 0.5f * length(b - a)};
    for (size_t i = 0; i < n; ++i) {
        const vec3 p = point(i);
        const float d = length(p - s.center);
        if (d > s.radius) {
            const float r = 0.5f * (s.radius + d);
            s.center = s.center + (p - s.center) * ((r - s.radius) / d);
            s.radius = r;
        }
    }
    s.radius *= 1.0001f;  // Rounding
    return s;
}

struct mat4 {
    float m[16] = {};  // Column major: m[col * 4 + row]
    float& at(int row, int col) { return m[col * 4 + row]; }
    float at(int row, int col) const { return m[col * 4 + row]; }
    static mat4 identity() {
        mat4 r;
        for (int i = 0; i < 4; ++i) r.at(i, i) = 1;
        return r;
    }
};

inline mat4 operator*(const mat4& a, const mat4& b) {
    mat4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.at(i, k) * b.at(k, j);
            r.at(i, j) = s;
        }
    return r;
}

inline vec3 transform_point(const mat4& a, vec3 p) {
    return {a.at(0, 0) * p.x + a.at(0, 1) * p.y + a.at(0, 2) * p.z + a.at(0, 3),
            a.at(1, 0) * p.x + a.at(1, 1) * p.y + a.at(1, 2) * p.z + a.at(1, 3),
            a.at(2, 0) * p.x + a.at(2, 1) * p.y + a.at(2, 2) * p.z + a.at(2, 3)};
}

// A camera looking from eye along forward, with up roughly up.
inline mat4 look_to(vec3 eye, vec3 forward, vec3 up) {
    const vec3 f = normalize(forward), s = normalize(cross(f, up)), u = cross(s, f);
    mat4 r = mat4::identity();
    r.at(0, 0) = s.x; r.at(0, 1) = s.y; r.at(0, 2) = s.z; r.at(0, 3) = -dot(s, eye);
    r.at(1, 0) = u.x; r.at(1, 1) = u.y; r.at(1, 2) = u.z; r.at(1, 3) = -dot(u, eye);
    r.at(2, 0) = -f.x; r.at(2, 1) = -f.y; r.at(2, 2) = -f.z; r.at(2, 3) = dot(f, eye);
    return r;
}

// Perspective with reversed depth and no far plane: depth is near / z, 1 at
// the near plane and falling toward 0, which spreads float precision evenly
// over distance. Vulkan's y runs down, so y is flipped here.
inline mat4 perspective_reverse_z(float fov_y, float aspect, float near_z) {
    const float f = 1 / std::tan(fov_y / 2);
    mat4 r;
    r.at(0, 0) = f / aspect;
    r.at(1, 1) = -f;
    r.at(2, 3) = near_z;
    r.at(3, 2) = -1;
    return r;
}

// Rotation by angle around a unit axis, then uniform scale, then translation.
inline mat4 trs(vec3 t, vec3 axis, float angle, float scale) {
    const float c = std::cos(angle), s = std::sin(angle), k = 1 - c;
    const vec3 a = normalize(axis);
    mat4 r = mat4::identity();
    r.at(0, 0) = (c + a.x * a.x * k) * scale;
    r.at(0, 1) = (a.x * a.y * k - a.z * s) * scale;
    r.at(0, 2) = (a.x * a.z * k + a.y * s) * scale;
    r.at(1, 0) = (a.y * a.x * k + a.z * s) * scale;
    r.at(1, 1) = (c + a.y * a.y * k) * scale;
    r.at(1, 2) = (a.y * a.z * k - a.x * s) * scale;
    r.at(2, 0) = (a.z * a.x * k - a.y * s) * scale;
    r.at(2, 1) = (a.z * a.y * k + a.x * s) * scale;
    r.at(2, 2) = (c + a.z * a.z * k) * scale;
    r.at(0, 3) = t.x; r.at(1, 3) = t.y; r.at(2, 3) = t.z;
    return r;
}

// The inverse of a general 4x4 matrix, by cofactors.
inline mat4 inverse(const mat4& a) {
    const float* m = a.m;
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    mat4 r;
    for (int i = 0; i < 16; ++i) r.m[i] = inv[i] / det;
    return r;
}
