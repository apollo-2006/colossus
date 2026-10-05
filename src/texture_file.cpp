#include "texture_file.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

constexpr char magic[8] = {'C', 'T', 'E', 'X', 'v', '0', '0', '2'};

float to_linear(uint8_t c) {
    const float s = c / 255.0f;
    return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
}
uint8_t to_srgb(float l) {
    const float s = l <= 0.0031308f ? l * 12.92f : 1.055f * std::pow(l, 1 / 2.4f) - 0.055f;
    return static_cast<uint8_t>(std::lround(std::clamp(s, 0.0f, 1.0f) * 255));
}

uint16_t pack565(const float* c) {
    auto q = [](float v, int max) { return static_cast<uint16_t>(std::lround(std::clamp(v, 0.0f, 255.0f) * max / 255)); };
    return q(c[0], 31) << 11 | q(c[1], 63) << 5 | q(c[2], 31);
}
void unpack565(uint16_t v, int* c) {
    c[0] = (v >> 11) * 255 / 31;
    c[1] = (v >> 5 & 63) * 255 / 63;
    c[2] = (v & 31) * 255 / 31;
}

// the four colours of a block, as bc1 decodes them (four-colour mode).
void palette(uint16_t c0, uint16_t c1, int (*p)[3]) {
    unpack565(c0, p[0]);
    unpack565(c1, p[1]);
    for (int k = 0; k < 3; ++k) {
        p[2][k] = (2 * p[0][k] + p[1][k]) / 3;
        p[3][k] = (p[0][k] + 2 * p[1][k]) / 3;
    }
}

// indices nearest each texel; their squared error.
long assign(const uint8_t* rgb, uint16_t c0, uint16_t c1, uint32_t& bits) {
    int p[4][3];
    palette(c0, c1, p);
    bits = 0;
    long total = 0;
    for (int i = 0; i < 16; ++i) {
        long best = LONG_MAX;
        int pick = 0;
        for (int k = 0; k < 4; ++k) {
            long d = 0;
            for (int c = 0; c < 3; ++c) d += long(rgb[3 * i + c] - p[k][c]) * (rgb[3 * i + c] - p[k][c]);
            if (d < best) { best = d; pick = k; }
        }
        bits |= uint32_t(pick) << (2 * i);
        total += best;
    }
    return total;
}

}  // namespace

// endpoints from the block's principal axis (its colours' extremes along it), then one least
// squares refit of the endpoints to the chosen indices, kept if better.
void encode_bc1(const uint8_t* rgb, uint8_t* out) {
    float mean[3] = {0, 0, 0};
    for (int i = 0; i < 16; ++i)
        for (int c = 0; c < 3; ++c) mean[c] += rgb[3 * i + c] / 16.0f;
    float cov[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 16; ++i) {
        const float d[3] = {rgb[3 * i] - mean[0], rgb[3 * i + 1] - mean[1], rgb[3 * i + 2] - mean[2]};
        cov[0] += d[0] * d[0]; cov[1] += d[0] * d[1]; cov[2] += d[0] * d[2];
        cov[3] += d[1] * d[1]; cov[4] += d[1] * d[2]; cov[5] += d[2] * d[2];
    }
    float axis[3] = {1, 1, 1};
    for (int it = 0; it < 8; ++it) {
        const float x = cov[0] * axis[0] + cov[1] * axis[1] + cov[2] * axis[2];
        const float y = cov[1] * axis[0] + cov[3] * axis[1] + cov[4] * axis[2];
        const float z = cov[2] * axis[0] + cov[4] * axis[1] + cov[5] * axis[2];
        const float l = std::max({std::abs(x), std::abs(y), std::abs(z)});
        if (l < 1e-6f) break;
        axis[0] = x / l; axis[1] = y / l; axis[2] = z / l;
    }
    float lo = INFINITY, hi = -INFINITY;
    for (int i = 0; i < 16; ++i) {
        const float t = (rgb[3 * i] - mean[0]) * axis[0] + (rgb[3 * i + 1] - mean[1]) * axis[1] + (rgb[3 * i + 2] - mean[2]) * axis[2];
        lo = std::min(lo, t);
        hi = std::max(hi, t);
    }
    const float l2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    float e0[3], e1[3];
    for (int c = 0; c < 3; ++c) {
        e0[c] = mean[c] + axis[c] * hi / l2;
        e1[c] = mean[c] + axis[c] * lo / l2;
    }
    auto finish = [](uint16_t& a, uint16_t& b) {
        // four-colour mode wants c0 > c1.
        if (a < b) std::swap(a, b);
    };
    uint16_t c0 = pack565(e0), c1 = pack565(e1);
    finish(c0, c1);
    uint32_t bits;
    long err = assign(rgb, c0, c1, bits);
    if (c0 != c1) {
        // least squares endpoints for these indices: texel = a * w + b * (1 - w).
        const float weight[4] = {1, 0, 2 / 3.0f, 1 / 3.0f};
        float aa = 0, ab = 0, bb = 0, ax[3] = {0, 0, 0}, bx[3] = {0, 0, 0};
        for (int i = 0; i < 16; ++i) {
            const float w = weight[bits >> (2 * i) & 3], v = 1 - w;
            aa += w * w; ab += w * v; bb += v * v;
            for (int c = 0; c < 3; ++c) { ax[c] += w * rgb[3 * i + c]; bx[c] += v * rgb[3 * i + c]; }
        }
        const float det = aa * bb - ab * ab;
        if (std::abs(det) > 1e-6f) {
            float a[3], b[3];
            for (int c = 0; c < 3; ++c) {
                a[c] = (ax[c] * bb - bx[c] * ab) / det;
                b[c] = (bx[c] * aa - ax[c] * ab) / det;
            }
            uint16_t r0 = pack565(a), r1 = pack565(b);
            finish(r0, r1);
            uint32_t rbits;
            const long rerr = r0 != r1 ? assign(rgb, r0, r1, rbits) : LONG_MAX;
            if (rerr < err) { c0 = r0; c1 = r1; bits = rbits; err = rerr; }
        }
    }
    if (c0 == c1) bits = 0;  // equal endpoints decode in three-colour mode: index 0 is still c0
    std::memcpy(out, &c0, 2);
    std::memcpy(out + 2, &c1, 2);
    std::memcpy(out + 4, &bits, 4);
}

void decode_bc1(const uint8_t* block, uint8_t* rgb) {
    uint16_t c0, c1;
    uint32_t bits;
    std::memcpy(&c0, block, 2);
    std::memcpy(&c1, block + 2, 2);
    std::memcpy(&bits, block + 4, 4);
    int p[4][3];
    palette(c0, c1, p);
    if (c0 <= c1)
        for (int k = 0; k < 3; ++k) {
            p[2][k] = (p[0][k] + p[1][k]) / 2;
            p[3][k] = 0;
        }
    for (int i = 0; i < 16; ++i)
        for (int c = 0; c < 3; ++c) rgb[3 * i + c] = static_cast<uint8_t>(p[bits >> (2 * i) & 3][c]);
}

image load_ppm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string tag;
    uint32_t max = 0;
    image im;
    auto skip = [&] {  // whitespace and comments
        for (;;) {
            const int c = f.peek();
            if (c == '#') { std::string line; std::getline(f, line); }
            else if (std::isspace(c)) f.get();
            else break;
        }
    };
    f >> tag;
    skip(); f >> im.width;
    skip(); f >> im.height;
    skip(); f >> max;
    f.get();
    if (!f || tag != "P6" || max != 255 || im.width == 0 || im.height == 0 || im.width > 65536 || im.height > 65536)
        throw std::runtime_error(path + ": want a binary 8-bit ppm");
    im.rgb.resize(size_t(im.width) * im.height * 3);
    f.read(reinterpret_cast<char*>(im.rgb.data()), static_cast<std::streamsize>(im.rgb.size()));
    if (!f) throw std::runtime_error(path + ": ends early");
    return im;
}

uint32_t texture_info::parent(uint32_t tile) const {
    for (const texture_entry& t : textures)
        for (size_t l = 0; l < t.levels.size(); ++l) {
            const texture_level& v = t.levels[l];
            if (tile < v.first_tile || tile >= v.first_tile + v.tiles_x * v.tiles_y) continue;
            if (l + 1 == t.levels.size()) return UINT32_MAX;
            const uint32_t i = tile - v.first_tile, x = i % v.tiles_x, y = i / v.tiles_x;
            const texture_level& up = t.levels[l + 1];
            return up.first_tile + std::min(y / 2, up.tiles_y - 1) * up.tiles_x + std::min(x / 2, up.tiles_x - 1);
        }
    return UINT32_MAX;
}

uint32_t texture_info::tile_count() const {
    if (textures.empty()) return 0;
    const texture_level& l = textures.back().levels.back();
    return l.first_tile + l.tiles_x * l.tiles_y;
}

namespace {

// a texture's levels: tile counts and first tiles from `first`, halved (rounded up) until one
// tile's payload holds one.
std::vector<texture_level> plan_levels(uint32_t w, uint32_t h, uint32_t first) {
    std::vector<texture_level> levels;
    for (;;) {
        const uint32_t tx = (w + tile_payload - 1) / tile_payload, ty = (h + tile_payload - 1) / tile_payload;
        levels.push_back({w, h, tx, ty, first});
        first += tx * ty;
        if (tx == 1 && ty == 1) return levels;
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
}

}  // namespace

void save_textures(const std::vector<image>& images, const std::vector<uint8_t>& flags, vec2 uv_min, float uv_extent,
                   const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(magic, sizeof magic);
    const uint32_t count = static_cast<uint32_t>(images.size());
    const float map[3] = {uv_min.x, uv_min.y, uv_extent};
    f.write(reinterpret_cast<const char*>(&count), 4);
    f.write(reinterpret_cast<const char*>(map), sizeof map);
    std::vector<std::vector<texture_level>> plans;
    uint32_t first = 0;
    for (size_t t = 0; t < images.size(); ++t) {
        plans.push_back(plan_levels(images[t].width, images[t].height, first));
        first = plans.back().back().first_tile + 1;
        const uint32_t head[4] = {images[t].width, images[t].height, uint32_t(plans.back().size()), uint32_t(flags[t])};
        f.write(reinterpret_cast<const char*>(head), sizeof head);
        for (const texture_level& l : plans.back()) {
            const uint32_t v[3] = {l.tiles_x, l.tiles_y, l.first_tile};
            f.write(reinterpret_cast<const char*>(v), sizeof v);
        }
    }
    while (f && f.tellp() % 16) f.put(0);  // a failed stream's tellp is -1

    std::vector<uint8_t> out;
    for (size_t t = 0; t < images.size(); ++t) {
        const image& base = images[t];
        // levels in linear light, box filtered (edges clamped).
        std::vector<float> linear(base.rgb.size());
        for (size_t i = 0; i < base.rgb.size(); ++i) linear[i] = to_linear(base.rgb[i]);
        uint32_t w = base.width, h = base.height;
        for (size_t li = 0; li < plans[t].size(); ++li) {
            const texture_level& l = plans[t][li];
            if (li > 0) {
                const uint32_t nw = (w + 1) / 2, nh = (h + 1) / 2;
                std::vector<float> dst(size_t(nw) * nh * 3);
                parallel_for(nh, [&](size_t y) {
                    for (uint32_t x = 0; x < nw; ++x)
                        for (int c = 0; c < 3; ++c) {
                            float sum = 0;
                            for (uint32_t dy = 0; dy < 2; ++dy)
                                for (uint32_t dx = 0; dx < 2; ++dx) {
                                    const uint32_t sx = std::min(2 * x + dx, w - 1), sy = std::min(uint32_t(2 * y + dy), h - 1);
                                    sum += linear[(size_t(sy) * w + sx) * 3 + c];
                                }
                            dst[(y * nw + x) * 3 + c] = sum / 4;
                        }
                });
                linear = std::move(dst);
                w = nw;
                h = nh;
            }
            std::vector<uint8_t> srgb(linear.size());
            for (size_t i = 0; i < linear.size(); ++i) srgb[i] = to_srgb(linear[i]);
            // each tile: 120 texels of the level and a 4 texel border, clamped at the level's
            // edges, or wrapped round them for a repeating texture (filtering across the
            // repeat then has its neighbours).
            const bool wrap = flags[t] & 1;
            auto at = [&](int x, int y) {
                if (wrap) {
                    x = ((x % int(w)) + int(w)) % int(w);
                    y = ((y % int(h)) + int(h)) % int(h);
                } else {
                    x = std::clamp(x, 0, int(w) - 1);
                    y = std::clamp(y, 0, int(h) - 1);
                }
                return &srgb[(size_t(y) * w + x) * 3];
            };
            out.assign(size_t(l.tiles_x) * l.tiles_y * tile_bytes, 0);
            parallel_for(size_t(l.tiles_x) * l.tiles_y, [&](size_t tile) {
                const int ox = int(tile % l.tiles_x) * int(tile_payload) - int(tile_border);
                const int oy = int(tile / l.tiles_x) * int(tile_payload) - int(tile_border);
                uint8_t block[48];
                for (uint32_t by = 0; by < tile_texels / 4; ++by)
                    for (uint32_t bx = 0; bx < tile_texels / 4; ++bx) {
                        for (int i = 0; i < 16; ++i) std::memcpy(block + 3 * i, at(ox + int(4 * bx) + i % 4, oy + int(4 * by) + i / 4), 3);
                        encode_bc1(block, &out[tile * tile_bytes + (by * (tile_texels / 4) + bx) * 8]);
                    }
            });
            f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
        }
    }
    if (!f) throw std::runtime_error("writing " + path + " failed");
}

void save_texture(const image& base, const std::string& path) { save_textures({base}, {0}, {0, 0}, 1, path); }

texture_info load_texture_info(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char m[sizeof magic];
    uint32_t count;
    float map[3];
    f.read(m, sizeof m);
    f.read(reinterpret_cast<char*>(&count), 4);
    f.read(reinterpret_cast<char*>(map), sizeof map);
    if (!f || std::memcmp(m, magic, sizeof magic) != 0) throw std::runtime_error(path + " is not a texture file (or an old one: rebuild it)");
    if (count == 0 || count > 256 || !(map[2] > 0)) throw std::runtime_error(path + ": bad header");
    texture_info t;
    t.uv_min = {map[0], map[1]};
    t.uv_extent = map[2];
    uint32_t first = 0;
    for (uint32_t k = 0; k < count; ++k) {
        uint32_t head[4];
        f.read(reinterpret_cast<char*>(head), sizeof head);
        if (!f || head[0] == 0 || head[1] == 0 || head[0] > 65536 || head[1] > 65536 || head[2] == 0 || head[2] > 20 || head[3] > 3)
            throw std::runtime_error(path + ": bad texture header");
        texture_entry e;
        e.width = head[0];
        e.height = head[1];
        e.repeat = head[3] & 1;
        e.double_sided = head[3] & 2;
        // the layout follows from the size: checked, not trusted.
        e.levels = plan_levels(e.width, e.height, first);
        if (e.levels.size() != head[2]) throw std::runtime_error(path + ": bad level count");
        for (const texture_level& l : e.levels) {
            uint32_t v[3];
            f.read(reinterpret_cast<char*>(v), sizeof v);
            if (!f || v[0] != l.tiles_x || v[1] != l.tiles_y || v[2] != l.first_tile) throw std::runtime_error(path + ": bad level table");
        }
        first = e.levels.back().first_tile + 1;
        t.textures.push_back(std::move(e));
    }
    while (f && f.tellg() % 16) f.get();  // past the end, tellg is -1
    t.data_offset = static_cast<uint64_t>(f.tellg());
    f.seekg(0, std::ios::end);
    if (!f || uint64_t(f.tellg()) < t.data_offset + uint64_t(t.tile_count()) * tile_bytes) throw std::runtime_error(path + ": ends early");
    return t;
}
