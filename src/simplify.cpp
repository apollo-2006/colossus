#include "simplify.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

namespace {

// A symmetric 4x4 quadric, the upper triangle of
//   [ A  b ]
//   [ b' c ]
// so that the error at p is p'Ap + 2b'p + c. Doubles: a sum over hundreds of
// nearly identical planes loses too much in float.
struct quadric {
    double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
    double b0 = 0, b1 = 0, b2 = 0, c = 0;
    double weight = 0;

    static quadric plane(double nx, double ny, double nz, double d, double w) {
        quadric q;
        q.a00 = w * nx * nx; q.a01 = w * nx * ny; q.a02 = w * nx * nz;
        q.a11 = w * ny * ny; q.a12 = w * ny * nz; q.a22 = w * nz * nz;
        q.b0 = w * nx * d; q.b1 = w * ny * d; q.b2 = w * nz * d;
        q.c = w * d * d;
        q.weight = w;
        return q;
    }
    quadric& operator+=(const quadric& o) {
        a00 += o.a00; a01 += o.a01; a02 += o.a02; a11 += o.a11; a12 += o.a12; a22 += o.a22;
        b0 += o.b0; b1 += o.b1; b2 += o.b2; c += o.c;
        weight += o.weight;
        return *this;
    }
    double error(vec3 p) const {
        const double x = p.x, y = p.y, z = p.z;
        const double e = x * (a00 * x + a01 * y + a02 * z) + y * (a01 * x + a11 * y + a12 * z) +
                         z * (a02 * x + a12 * y + a22 * z) + 2 * (b0 * x + b1 * y + b2 * z) + c;
        return std::max(0.0, e);
    }
};

struct collapse {
    double cost;
    uint32_t from, to;
    uint32_t from_version, to_version;
    bool operator>(const collapse& o) const { return cost > o.cost; }
};

}  // namespace

simplify_result simplify(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices,
                         const std::vector<uint8_t>& locked, size_t target_triangles) {
    simplify_result out;
    const size_t n = indices.size() / 3;
    if (n <= target_triangles) {
        out.indices = indices;
        return out;
    }

    // Local vertex numbers.
    std::vector<uint32_t> verts(indices);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    const size_t nv = verts.size();
    std::vector<uint32_t> tri(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
        tri[i] = static_cast<uint32_t>(std::lower_bound(verts.begin(), verts.end(), indices[i]) - verts.begin());
    std::vector<vec3> pos(nv);
    std::vector<uint8_t> fixed(nv);
    for (size_t v = 0; v < nv; ++v) {
        pos[v] = positions[verts[v]];
        fixed[v] = locked[verts[v]];
    }

    std::vector<std::vector<uint32_t>> vertex_tris(nv);
    for (size_t t = 0; t < n; ++t)
        for (int c = 0; c < 3; ++c) vertex_tris[tri[3 * t + c]].push_back(static_cast<uint32_t>(t));

    auto face_normal = [&](size_t t) {  // Not normalized: length is twice the area
        const vec3 a = pos[tri[3 * t]], b = pos[tri[3 * t + 1]], c = pos[tri[3 * t + 2]];
        return cross(b - a, c - a);
    };

    std::vector<quadric> q(nv);
    for (size_t t = 0; t < n; ++t) {
        const vec3 fn = face_normal(t);
        const double len = length(fn);
        if (len <= 0) continue;
        const double nx = fn.x / len, ny = fn.y / len, nz = fn.z / len;
        const vec3 p = pos[tri[3 * t]];
        const quadric plane = quadric::plane(nx, ny, nz, -(nx * p.x + ny * p.y + nz * p.z), 0.5 * len);
        for (int c = 0; c < 3; ++c) q[tri[3 * t + c]] += plane;
    }

    // Edges, each once, with how many triangles use them. An edge used once
    // is an open border: a plane through it, perpendicular to its triangle,
    // keeps collapses from dragging the border inward.
    struct edge { uint32_t a, b, uses; size_t tri; };
    std::vector<edge> edges;
    edges.reserve(3 * n);
    for (size_t t = 0; t < n; ++t)
        for (int c = 0; c < 3; ++c) {
            uint32_t a = tri[3 * t + c], b = tri[3 * t + (c + 1) % 3];
            if (a > b) std::swap(a, b);
            edges.push_back({a, b, 1, t});
        }
    std::sort(edges.begin(), edges.end(), [](const edge& x, const edge& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
    {
        size_t w = 0;
        for (size_t i = 0; i < edges.size(); ++i) {
            if (w > 0 && edges[w - 1].a == edges[i].a && edges[w - 1].b == edges[i].b) ++edges[w - 1].uses;
            else edges[w++] = edges[i];
        }
        edges.resize(w);
    }
    std::vector<uint8_t> border(nv, 0);
    for (const edge& e : edges) {
        if (e.uses != 1) continue;
        border[e.a] = border[e.b] = 1;
        const vec3 fn = normalize(face_normal(e.tri));
        const vec3 along = pos[e.b] - pos[e.a];
        const vec3 n_border = normalize(cross(along, fn));
        if (dot(n_border, n_border) == 0) continue;
        const float d = -dot(n_border, pos[e.a]);
        // Weighted like a triangle with the edge's length on each side.
        const quadric plane = quadric::plane(n_border.x, n_border.y, n_border.z, d, dot(along, along));
        q[e.a] += plane;
        q[e.b] += plane;
    }

    std::vector<uint32_t> version(nv, 0);
    std::vector<uint8_t> tri_alive(n, 1);
    std::priority_queue<collapse, std::vector<collapse>, std::greater<collapse>> heap;

    auto cost_of = [&](uint32_t from, uint32_t to) {
        quadric sum = q[from];
        sum += q[to];
        return sum.error(pos[to]) / std::max(sum.weight, 1e-30);
    };
    auto push_edge = [&](uint32_t a, uint32_t b) {
        const bool ab = !fixed[a], ba = !fixed[b];
        if (!ab && !ba) return;
        const double cab = ab ? cost_of(a, b) : INFINITY, cba = ba ? cost_of(b, a) : INFINITY;
        if (cab <= cba) heap.push({cab, a, b, version[a], version[b]});
        else heap.push({cba, b, a, version[b], version[a]});
    };
    for (const edge& e : edges) push_edge(e.a, e.b);

    std::vector<uint32_t> mark(nv, 0);
    uint32_t mark_id = 0;
    size_t live = n;
    double worst = 0;
    std::vector<uint32_t> neighbours;

    while (live > target_triangles && !heap.empty()) {
        const collapse c = heap.top();
        heap.pop();
        const uint32_t from = c.from, to = c.to;
        if (version[from] != c.from_version || version[to] != c.to_version) continue;

        // Triangles on the edge, and the link condition: the vertices next
        // to both ends must be exactly the far corners of those triangles,
        // or the collapse would pinch the surface.
        ++mark_id;
        uint32_t shared = 0;
        for (uint32_t t : vertex_tris[from]) {
            if (!tri_alive[t]) continue;
            bool has_to = false;
            for (int k = 0; k < 3; ++k) has_to |= tri[3 * t + k] == to;
            shared += has_to;
            for (int k = 0; k < 3; ++k) mark[tri[3 * t + k]] = mark_id;
        }
        if (shared == 0) continue;  // The edge is gone
        // A vertex on an open border may only slide along it, onto the next
        // border vertex: anything else would pull the hole's outline inward
        // or pinch it.
        if (border[from] && shared != 1) continue;
        // Nor may a collapse delete a piece of surface outright: if no
        // triangle would be left at `to`, an island of the scan is gone.
        {
            uint32_t at_to = 0;
            for (uint32_t t : vertex_tris[to]) at_to += tri_alive[t];
            uint32_t at_from = 0;
            for (uint32_t t : vertex_tris[from]) at_from += tri_alive[t];
            if (at_to + at_from - 2 * shared == 0) continue;
        }
        uint32_t common = 0;
        ++mark_id;
        const uint32_t from_mark = mark_id - 1;
        for (uint32_t t : vertex_tris[to]) {
            if (!tri_alive[t]) continue;
            for (int k = 0; k < 3; ++k) {
                const uint32_t v = tri[3 * t + k];
                if (v != from && v != to && mark[v] == from_mark) {
                    mark[v] = mark_id;  // Count each once
                    ++common;
                }
            }
        }
        if (common != shared) continue;

        // Moving from onto to must not flip or flatten any triangle that
        // stays.
        bool ok = true;
        for (uint32_t t : vertex_tris[from]) {
            if (!tri_alive[t]) continue;
            const uint32_t* v = &tri[3 * t];
            if (v[0] == to || v[1] == to || v[2] == to) continue;
            const vec3 before = face_normal(t);
            vec3 p[3];
            for (int k = 0; k < 3; ++k) p[k] = v[k] == from ? pos[to] : pos[v[k]];
            const vec3 after = cross(p[1] - p[0], p[2] - p[0]);
            const float la = length(after), lb = length(before);
            if (la <= 1e-6f * lb || dot(after, before) <= 0.25f * la * lb) { ok = false; break; }
        }
        if (!ok) continue;

        worst = std::max(worst, c.cost);
        for (uint32_t t : vertex_tris[from]) {
            if (!tri_alive[t]) continue;
            uint32_t* v = &tri[3 * t];
            if (v[0] == to || v[1] == to || v[2] == to) {
                tri_alive[t] = 0;
                --live;
            } else {
                for (int k = 0; k < 3; ++k)
                    if (v[k] == from) v[k] = to;
                vertex_tris[to].push_back(t);
            }
        }
        vertex_tris[from].clear();
        q[to] += q[from];
        ++version[from];
        ++version[to];

        // Every edge at to changed cost; requeue them.
        auto& tt = vertex_tris[to];
        tt.erase(std::remove_if(tt.begin(), tt.end(), [&](uint32_t t) { return !tri_alive[t]; }), tt.end());
        neighbours.clear();
        for (uint32_t t : tt)
            for (int k = 0; k < 3; ++k)
                if (tri[3 * t + k] != to) neighbours.push_back(tri[3 * t + k]);
        std::sort(neighbours.begin(), neighbours.end());
        neighbours.erase(std::unique(neighbours.begin(), neighbours.end()), neighbours.end());
        for (uint32_t x : neighbours) push_edge(to, x);
    }

    out.indices.reserve(3 * live);
    for (size_t t = 0; t < n; ++t)
        if (tri_alive[t])
            for (int k = 0; k < 3; ++k) out.indices.push_back(verts[tri[3 * t + k]]);
    out.error = static_cast<float>(std::sqrt(worst));
    return out;
}
