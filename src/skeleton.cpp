#include "skeleton.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

constexpr char magic[8] = {'C', 'S', 'K', 'N', 'v', '0', '0', '1'};

template <class T>
void put(std::ofstream& f, const T& v) { f.write(reinterpret_cast<const char*>(&v), sizeof v); }
template <class T>
void put_vec(std::ofstream& f, const std::vector<T>& v) {
    put(f, uint64_t(v.size()));
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
}
template <class T>
void get(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof v);
    if (!f) throw std::runtime_error("skeleton file: ends early");
}
template <class T>
void get_vec(std::ifstream& f, std::vector<T>& v) {
    uint64_t n;
    get(f, n);
    if (n > (uint64_t(1) << 28) / sizeof(T)) throw std::runtime_error("skeleton file: bad array length");
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
    if (!f) throw std::runtime_error("skeleton file: ends early");
}

// column major 4x4.
struct m4 {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};
m4 mul(const m4& a, const m4& b) {
    m4 r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}
m4 trs(const float* t, const float* q, const float* s) {
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float rot[9] = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w),     2 * (x * z - y * w),
                          2 * (x * y - z * w),     1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
                          2 * (x * z + y * w),     2 * (y * z - x * w),     1 - 2 * (x * x + y * y)};
    m4 r;
    for (int c = 0; c < 3; ++c)
        for (int row = 0; row < 3; ++row) r.m[c * 4 + row] = rot[c * 3 + row] * s[c];
    r.m[12] = t[0]; r.m[13] = t[1]; r.m[14] = t[2];
    return r;
}

// a channel's value at time t: linear between keys (rotations normalized), held past the ends.
void sample(const animation_channel& c, float t, float* out) {
    const int w = c.path == 1 ? 4 : 3;
    const auto& ts = c.times;
    size_t k = size_t(std::upper_bound(ts.begin(), ts.end(), t) - ts.begin());
    if (k == 0 || ts.size() == 1) { std::copy_n(&c.values[0], w, out); return; }
    if (k >= ts.size()) { std::copy_n(&c.values[(ts.size() - 1) * w], w, out); return; }
    const float span = ts[k] - ts[k - 1];
    const float f = c.step || span <= 0 ? 0.0f : (t - ts[k - 1]) / span;
    const float* a = &c.values[(k - 1) * w];
    const float* b = &c.values[k * w];
    float sign = 1;
    if (w == 4) {  // the shorter way round
        float d = 0;
        for (int i = 0; i < 4; ++i) d += a[i] * b[i];
        if (d < 0) sign = -1;
    }
    for (int i = 0; i < w; ++i) out[i] = a[i] + (sign * b[i] - a[i]) * f;
    if (w == 4) {
        const float l = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]);
        for (int i = 0; i < 4; ++i) out[i] /= std::max(l, 1e-30f);
    }
}

}  // namespace

void save_skeleton(const skeleton& s, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(magic, sizeof magic);
    put(f, s.place);
    put(f, s.anchor);
    put_vec(f, s.nodes);
    put_vec(f, s.joints);
    put_vec(f, s.inverse_bind);
    put_vec(f, s.pages);
    put(f, uint64_t(s.animations.size()));
    for (const animation& a : s.animations) {
        std::vector<char> name(a.name.begin(), a.name.end());
        put_vec(f, name);
        put(f, a.duration);
        put(f, uint64_t(a.channels.size()));
        for (const animation_channel& c : a.channels) {
            put(f, c.node);
            put(f, c.path);
            put(f, c.step);
            put_vec(f, c.times);
            put_vec(f, c.values);
        }
    }
    if (!f) throw std::runtime_error("writing " + path + " failed");
}

skeleton load_skeleton(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char m[sizeof magic];
    f.read(m, sizeof m);
    if (!f || std::memcmp(m, magic, sizeof magic) != 0) throw std::runtime_error(path + " is not a skeleton file (or an old one: rebuild it)");
    skeleton s;
    get(f, s.place);
    get(f, s.anchor);
    get_vec(f, s.nodes);
    get_vec(f, s.joints);
    get_vec(f, s.inverse_bind);
    get_vec(f, s.pages);
    uint64_t anims;
    get(f, anims);
    if (anims > 4096) throw std::runtime_error(path + ": too many animations");
    for (uint64_t i = 0; i < anims; ++i) {
        animation a;
        std::vector<char> name;
        get_vec(f, name);
        a.name.assign(name.begin(), name.end());
        get(f, a.duration);
        uint64_t channels;
        get(f, channels);
        if (channels > 65536) throw std::runtime_error(path + ": too many channels");
        for (uint64_t k = 0; k < channels; ++k) {
            animation_channel c;
            get(f, c.node);
            get(f, c.path);
            get(f, c.step);
            get_vec(f, c.times);
            get_vec(f, c.values);
            const size_t w = c.path == 1 ? 4 : 3;
            if (c.node >= s.nodes.size() || c.path > 2 || c.times.empty() || c.values.size() != c.times.size() * w)
                throw std::runtime_error(path + ": bad animation channel");
            a.channels.push_back(std::move(c));
        }
        s.animations.push_back(std::move(a));
    }
    // a tree: every parent in range, no cycles (each chain shorter than the node count).
    for (size_t i = 0; i < s.nodes.size(); ++i) {
        int p = s.nodes[i].parent;
        for (size_t steps = 0; p >= 0; ++steps) {
            if (size_t(p) >= s.nodes.size() || steps > s.nodes.size()) throw std::runtime_error(path + ": bad node tree");
            p = s.nodes[size_t(p)].parent;
        }
    }
    if (s.joints.size() > 64 || s.inverse_bind.size() != 16 * s.joints.size() || (s.anchor >= s.joints.size() && !s.joints.empty()))
        throw std::runtime_error(path + ": bad joints");
    for (uint32_t j : s.joints)
        if (j >= s.nodes.size()) throw std::runtime_error(path + ": joint out of range");
    return s;
}

void pose_joints(const skeleton& s, int animation_index, float t, std::vector<float>& rows) {
    const size_t n = s.nodes.size();
    std::vector<float> trsv(10 * n);  // translation, rotation, scale per node
    for (size_t i = 0; i < n; ++i) {
        std::copy_n(s.nodes[i].translation, 3, &trsv[10 * i]);
        std::copy_n(s.nodes[i].rotation, 4, &trsv[10 * i + 3]);
        std::copy_n(s.nodes[i].scale, 3, &trsv[10 * i + 7]);
    }
    if (animation_index >= 0 && size_t(animation_index) < s.animations.size()) {
        const animation& a = s.animations[size_t(animation_index)];
        const float at = a.duration > 0 ? std::fmod(std::fmod(t, a.duration) + a.duration, a.duration) : 0.0f;
        for (const animation_channel& c : a.channels) sample(c, at, &trsv[10 * c.node + (c.path == 0 ? 0 : c.path == 1 ? 3 : 7)]);
    }
    std::vector<m4> world(n);
    std::vector<uint8_t> done(n, 0);
    auto world_of = [&](auto&& self, size_t i) -> const m4& {
        if (!done[i]) {
            const m4 local = trs(&trsv[10 * i], &trsv[10 * i + 3], &trsv[10 * i + 7]);
            world[i] = s.nodes[i].parent < 0 ? local : mul(self(self, size_t(s.nodes[i].parent)), local);
            done[i] = 1;
        }
        return world[i];
    };
    // placement as a matrix and its inverse: p -> scale (turned p - base).
    m4 place, unplace;
    const placement& pl = s.place;
    if (pl.up_z) {  // (x, y, z) -> (x, z, -y)
        place.m[5] = 0; place.m[6] = -1; place.m[9] = 1; place.m[10] = 0;
        unplace.m[5] = 0; unplace.m[6] = 1; unplace.m[9] = -1; unplace.m[10] = 0;
    }
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) place.m[c * 4 + r] *= pl.scale;
    place.m[12] = -pl.base.x * pl.scale; place.m[13] = -pl.base.y * pl.scale; place.m[14] = -pl.base.z * pl.scale;
    // unplace: turned back after (q / scale + base).
    m4 unscale;
    for (int k = 0; k < 3; ++k) unscale.m[5 * k] = 1 / pl.scale;
    unscale.m[12] = pl.base.x; unscale.m[13] = pl.base.y; unscale.m[14] = pl.base.z;
    unplace = mul(unplace, unscale);
    rows.assign(12 * s.joints.size(), 0.0f);
    for (size_t j = 0; j < s.joints.size(); ++j) {
        m4 ibm;
        std::copy_n(&s.inverse_bind[16 * j], 16, ibm.m);
        const m4 jm = mul(place, mul(mul(world_of(world_of, s.joints[j]), ibm), unplace));
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) rows[12 * j + 4 * r + c] = jm.m[c * 4 + r];
    }
}
