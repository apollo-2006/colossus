#include "deviation.hpp"

#include <algorithm>
#include <cmath>

float point_triangle_distance(vec3 p, vec3 a, vec3 b, vec3 c) {
    const vec3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return length(p - a);
    const vec3 bp = p - b;
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return length(p - b);
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return length(p - (a + ab * (d1 / (d1 - d3))));
    const vec3 cp = p - c;
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return length(p - c);
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return length(p - (a + ac * (d2 / (d2 - d6))));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        return length(p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))));
    const float denom = 1 / (va + vb + vc);
    return length(p - (a + ab * (vb * denom) + ac * (vc * denom)));
}

namespace {

// triangles binned in a grid over their box, for nearest queries.
struct triangle_grid {
    const std::vector<vec3>& pos;
    const std::vector<uint32_t>& idx;
    vec3 lo;
    float cell = 1;
    int n[3] = {1, 1, 1};
    std::vector<std::vector<uint32_t>> cells;

    triangle_grid(const std::vector<vec3>& p, const std::vector<uint32_t>& i) : pos(p), idx(i) {
        vec3 hi;
        lo = hi = pos[idx[0]];
        for (uint32_t v : idx) { lo = min(lo, pos[v]); hi = max(hi, pos[v]); }
        const vec3 size = hi - lo;
        const float longest = std::max({size.x, size.y, size.z, 1e-20f});
        // about two triangles a cell, at most 24 cells a side.
        const int side = std::clamp(static_cast<int>(std::cbrt(idx.size() / 3 / 2.0)) + 1, 1, 24);
        cell = longest / side * 1.0001f;
        for (int a = 0; a < 3; ++a) n[a] = std::max(1, static_cast<int>(std::ceil(size[a] / cell)));
        cells.resize(size_t(n[0]) * n[1] * n[2]);
        for (uint32_t t = 0; t < idx.size() / 3; ++t) {
            vec3 tlo = pos[idx[3 * t]], thi = tlo;
            for (int k = 1; k < 3; ++k) { tlo = min(tlo, pos[idx[3 * t + k]]); thi = max(thi, pos[idx[3 * t + k]]); }
            int c0[3], c1[3];
            for (int a = 0; a < 3; ++a) { c0[a] = coord(tlo[a], a); c1[a] = coord(thi[a], a); }
            for (int z = c0[2]; z <= c1[2]; ++z)
                for (int y = c0[1]; y <= c1[1]; ++y)
                    for (int x = c0[0]; x <= c1[0]; ++x) cells[(size_t(z) * n[1] + y) * n[0] + x].push_back(t);
        }
    }

    int coord(float v, int a) const { return std::clamp(static_cast<int>((v - lo[a]) / cell), 0, n[a] - 1); }

    // distance to the nearest triangle: rings of cells outward until a ring is
    // farther than the best.
    float nearest(vec3 p) const {
        float best = INFINITY;
        int c[3];
        for (int a = 0; a < 3; ++a) c[a] = coord(p[a], a);
        const int limit = std::max({n[0], n[1], n[2]});
        for (int r = 0; r <= limit; ++r) {
            if (best <= (r - 1) * cell) break;
            for (int z = c[2] - r; z <= c[2] + r; ++z)
                for (int y = c[1] - r; y <= c[1] + r; ++y)
                    for (int x = c[0] - r; x <= c[0] + r; ++x) {
                        if (std::max({std::abs(x - c[0]), std::abs(y - c[1]), std::abs(z - c[2])}) != r) continue;
                        if (x < 0 || y < 0 || z < 0 || x >= n[0] || y >= n[1] || z >= n[2]) continue;
                        for (uint32_t t : cells[(size_t(z) * n[1] + y) * n[0] + x])
                            best = std::min(best, point_triangle_distance(p, pos[idx[3 * t]], pos[idx[3 * t + 1]], pos[idx[3 * t + 2]]));
                    }
        }
        return best;
    }
};

// every vertex and edge midpoint once, and every triangle centre.
float one_way(const std::vector<vec3>& pos, const std::vector<uint32_t>& from, const triangle_grid& to) {
    std::vector<uint32_t> verts(from);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    std::vector<uint64_t> edges;
    edges.reserve(from.size());
    for (size_t t = 0; t < from.size(); t += 3)
        for (int k = 0; k < 3; ++k) {
            const uint32_t a = from[t + k], b = from[t + (k + 1) % 3];
            edges.push_back(uint64_t(std::min(a, b)) << 32 | std::max(a, b));
        }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    float worst = 0;
    for (uint32_t v : verts) worst = std::max(worst, to.nearest(pos[v]));
    for (uint64_t e : edges) worst = std::max(worst, to.nearest((pos[e >> 32] + pos[e & 0xffffffffu]) * 0.5f));
    for (size_t t = 0; t < from.size(); t += 3)
        worst = std::max(worst, to.nearest((pos[from[t]] + pos[from[t + 1]] + pos[from[t + 2]]) * (1.0f / 3)));
    return worst;
}

}  // namespace

float mesh_deviation(const std::vector<vec3>& positions, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return 0;
    const triangle_grid ga(positions, a), gb(positions, b);
    return std::max(one_way(positions, a, gb), one_way(positions, b, ga));
}
