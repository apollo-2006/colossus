#include "deviation.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace {

float squared(vec3 v) { return dot(v, v); }

// the squared distance: comparisons need no square root.
float point_triangle_distance_squared(vec3 p, vec3 a, vec3 b, vec3 c) {
    const vec3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return squared(p - a);
    const vec3 bp = p - b;
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return squared(p - b);
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return squared(p - (a + ab * (d1 / (d1 - d3))));
    const vec3 cp = p - c;
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return squared(p - c);
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return squared(p - (a + ac * (d2 / (d2 - d6))));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        return squared(p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))));
    const float denom = 1 / (va + vb + vc);
    return squared(p - (a + ab * (vb * denom) + ac * (vc * denom)));
}

}  // namespace

float point_triangle_distance(vec3 p, vec3 a, vec3 b, vec3 c) {
    return std::sqrt(point_triangle_distance_squared(p, a, b, c));
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
    // each triangle's bounding sphere: a query skips a triangle whose sphere is farther than the
    // best found.
    std::vector<vec3> centres;
    std::vector<float> radii;
    // triangles around each vertex, for local searches.
    std::unordered_map<uint32_t, std::vector<uint32_t>> around;

    triangle_grid(const std::vector<vec3>& p, const std::vector<uint32_t>& i) : pos(p), idx(i) {
        for (uint32_t t = 0; t < idx.size() / 3; ++t) {
            for (int k = 0; k < 3; ++k) around[idx[3 * t + k]].push_back(t);
            const vec3 a = pos[idx[3 * t]], b = pos[idx[3 * t + 1]], c = pos[idx[3 * t + 2]];
            const vec3 centre = (a + b + c) * (1.0f / 3);
            centres.push_back(centre);
            radii.push_back(std::sqrt(std::max({squared(a - centre), squared(b - centre), squared(c - centre)})) * 1.0001f);
        }
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
    float nearest(vec3 p, uint32_t* which = nullptr) const {
        float best = INFINITY, best_squared = INFINITY;
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
                        for (uint32_t t : cells[(size_t(z) * n[1] + y) * n[0] + x]) {
                            const float reach = best + radii[t];
                            if (squared(p - centres[t]) > reach * reach) continue;  // its sphere is farther
                            const float d = point_triangle_distance_squared(p, pos[idx[3 * t]], pos[idx[3 * t + 1]], pos[idx[3 * t + 2]]);
                            if (d < best_squared) {
                                best_squared = d;
                                best = std::sqrt(d);
                                if (which) *which = t;
                            }
                        }
                    }
        }
        return best;
    }

    // distance to the nearest of the seeds' triangles and those around their corners: at least
    // the true distance (so any bound built on it holds), and usually it.
    float nearest_local(vec3 p, const uint32_t (&seeds)[3], uint32_t& which) const {
        float best = INFINITY, best_squared = INFINITY;
        for (uint32_t s : seeds)
            for (int k = 0; k < 3; ++k)
                for (uint32_t t : around.at(idx[3 * s + k])) {
                    const float reach = best + radii[t];
                    if (squared(p - centres[t]) > reach * reach) continue;  // its sphere is farther
                    const float d = point_triangle_distance_squared(p, pos[idx[3 * t]], pos[idx[3 * t + 1]], pos[idx[3 * t + 2]]);
                    if (d < best_squared) {
                        best_squared = d;
                        best = std::sqrt(d);
                        which = t;
                    }
                }
        return best;
    }

    float distance_to(vec3 p, uint32_t t) const {
        return point_triangle_distance(p, pos[idx[3 * t]], pos[idx[3 * t + 1]], pos[idx[3 * t + 2]]);
    }
};

// each vertex of a mesh, sorted, with its distance to the other and the triangle there.
struct vertex_nearest {
    std::vector<uint32_t> verts;
    std::vector<float> d;
    std::vector<uint32_t> near;
    size_t at(uint32_t v) const { return size_t(std::lower_bound(verts.begin(), verts.end(), v) - verts.begin()); }
};

// every vertex and edge midpoint once, and every triangle centre. the vertices' nearest go
// to `out` if given.
float one_way(const std::vector<vec3>& pos, const std::vector<uint32_t>& from, const triangle_grid& to, vertex_nearest* out = nullptr) {
    vertex_nearest local;
    vertex_nearest& vn = out ? *out : local;
    std::vector<uint32_t>& verts = vn.verts;
    verts = from;
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
    vn.d.resize(verts.size());
    vn.near.resize(verts.size());
    for (size_t k = 0; k < verts.size(); ++k) {
        vn.d[k] = to.nearest(pos[verts[k]], &vn.near[k]);
        worst = std::max(worst, vn.d[k]);
    }
    for (uint64_t e : edges) worst = std::max(worst, to.nearest((pos[e >> 32] + pos[e & 0xffffffffu]) * 0.5f));
    for (size_t t = 0; t < from.size(); t += 3)
        worst = std::max(worst, to.nearest((pos[from[t]] + pos[from[t + 1]] + pos[from[t + 2]]) * (1.0f / 3)));
    return worst;
}

// an upper bound on the largest distance from `from` to `to`, starting from the sampled one
// (`worst`, a lower bound).
float bounded_one_way(const std::vector<vec3>& pos, const std::vector<uint32_t>& from, const triangle_grid& to,
                      const vertex_nearest& vn, float worst, float slack, float tolerance) {
    struct piece {
        vec3 p[3];
        float d[3];
        uint32_t near[3];  // each corner's nearest triangle
        int depth;
    };
    // over a piece, the distance to one triangle is convex, so largest at a corner, and the
    // distance to the mesh at most that: the corners' nearest triangles are the candidates.
    // a corner's distance to its own nearest is known, and a triangle nearest two corners is
    // tried once.
    auto piece_bound = [&](const piece& q) {
        float best = INFINITY;
        for (int j = 0; j < 3; ++j) {
            const uint32_t t = q.near[j];
            if ((j > 0 && t == q.near[0]) || (j == 2 && t == q.near[1])) continue;
            float worst = 0;
            for (int k = 0; k < 3 && worst < best; ++k) worst = std::max(worst, q.near[k] == t ? q.d[k] : to.distance_to(q.p[k], t));
            best = std::min(best, worst);
        }
        return best;
    };
    constexpr int max_depth = 6;  // 4096 pieces a triangle at most
    const float spread = 1 / std::sqrt(3.0f);  // farthest from the nearest corner, per longest edge
    float bound = worst;
    std::vector<piece> stack;
    for (size_t t = 0; t < from.size(); t += 3) {
        piece first{};
        for (int k = 0; k < 3; ++k) {
            first.p[k] = pos[from[t + k]];
            const size_t at = vn.at(from[t + k]);
            first.d[k] = vn.d[at];
            first.near[k] = vn.near[at];
        }
        stack.push_back(first);
        while (!stack.empty()) {
            const piece q = stack.back();
            stack.pop_back();
            const float longest = std::max({length(q.p[1] - q.p[0]), length(q.p[2] - q.p[1]), length(q.p[0] - q.p[2])});
            float upper = std::max({q.d[0], q.d[1], q.d[2]}) + longest * spread;
            if (upper > worst * (1 + slack) && upper > tolerance) upper = std::min(upper, piece_bound(q));
            if (upper <= worst * (1 + slack) || upper <= tolerance || q.depth == max_depth) {
                bound = std::max(bound, upper);
                continue;
            }
            // four halves: corners and midpoints, each sampled once.
            vec3 m[3];
            float dm[3];
            uint32_t nm[3];
            for (int k = 0; k < 3; ++k) {
                m[k] = (q.p[k] + q.p[(k + 1) % 3]) * 0.5f;
                // the local search, unless it finds the midpoint farther than the corners: then the
                // true nearest may lie outside it, and an overestimate would raise `worst`.
                dm[k] = to.nearest_local(m[k], q.near, nm[k]);
                if (dm[k] > std::max({q.d[0], q.d[1], q.d[2]})) dm[k] = to.nearest(m[k], &nm[k]);
                worst = std::max(worst, dm[k]);
            }
            const int d = q.depth + 1;
            stack.push_back({{q.p[0], m[0], m[2]}, {q.d[0], dm[0], dm[2]}, {q.near[0], nm[0], nm[2]}, d});
            stack.push_back({{m[0], q.p[1], m[1]}, {dm[0], q.d[1], dm[1]}, {nm[0], q.near[1], nm[1]}, d});
            stack.push_back({{m[2], m[1], q.p[2]}, {dm[2], dm[1], q.d[2]}, {nm[2], nm[1], q.near[2]}, d});
            stack.push_back({{m[0], m[1], m[2]}, {dm[0], dm[1], dm[2]}, {nm[0], nm[1], nm[2]}, d});
        }
    }
    return std::max(bound, worst);
}

}  // namespace

float sampled_deviation(const std::vector<vec3>& positions, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return 0;
    const triangle_grid ga(positions, a), gb(positions, b);
    return std::max(one_way(positions, a, gb), one_way(positions, b, ga));
}

float mesh_deviation(const std::vector<vec3>& positions, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                     float slack, float tolerance) {
    if (a.empty() || b.empty()) return 0;
    const triangle_grid ga(positions, a), gb(positions, b);
    // the samples first: a high lower bound lets most triangles stop at once.
    vertex_nearest na, nb;
    const float sampled = std::max(one_way(positions, a, gb, &na), one_way(positions, b, ga, &nb));
    return std::max(bounded_one_way(positions, a, gb, na, sampled, slack, tolerance),
                    bounded_one_way(positions, b, ga, nb, sampled, slack, tolerance));
}
