#include "simplify.hpp"

#include <algorithm>
#include <functional>
#include <cmath>
#include <unordered_map>
#include <queue>

namespace {

// symmetric 4x4 quadric, the upper triangle of
//     [ A  b ]
//     [ b' c ]
// error at p is p'Ap + 2b'p + c. doubles: summing hundreds of near-equal planes
// loses too much in float.
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
                         const std::vector<uint8_t>& locked, size_t target_triangles, const wedges* w,
                         const std::vector<std::vector<vec3>>* poses, const std::vector<vec3>* normals, float normal_weight) {
    simplify_result out;
    const size_t n = indices.size() / 3;
    if (n <= target_triangles) {
        out.indices = indices;
        if (w) out.corners = w->corners;
        return out;
    }

    // local vertex numbers.
    std::vector<uint32_t> verts(indices);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    const size_t nv = verts.size();
    std::vector<uint32_t> tri(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
        tri[i] = static_cast<uint32_t>(std::lower_bound(verts.begin(), verts.end(), indices[i]) - verts.begin());
    std::vector<vec3> pos(nv), nrm(normals && normal_weight > 0 ? nv : 0);
    std::vector<uint8_t> fixed(nv);
    for (size_t v = 0; v < nv; ++v) {
        pos[v] = positions[verts[v]];
        fixed[v] = locked[verts[v]];
        if (!nrm.empty()) nrm[v] = (*normals)[verts[v]];
    }
    const double normal_weight2 = double(normal_weight) * normal_weight;

    // per corner its wedge; per vertex its wedges by chart (a vertex keeps its own as others
    // collapse onto it).
    std::vector<uint32_t> corner(w ? w->corners : std::vector<uint32_t>());
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> chart_wedges(w ? nv : 0);  // (chart, wedge)
    if (w)
        for (size_t i = 0; i < indices.size(); ++i) {
            auto& list = chart_wedges[tri[i]];
            const uint32_t chart = w->chart[corner[i]];
            bool have = false;
            for (const auto& [c, x] : list) have |= c == chart;
            if (!have) list.push_back({chart, corner[i]});
        }
    auto chart_of = [&](uint32_t id) {
        return id & new_wedge_bit ? out.new_wedges[id & ~new_wedge_bit].chart : w->chart[id];
    };
    auto uv_of = [&](uint32_t id) { return id & new_wedge_bit ? out.new_wedges[id & ~new_wedge_bit].uv : w->uv[id]; };
    auto material_of = [&](uint32_t id) { return id & new_wedge_bit ? out.new_wedges[id & ~new_wedge_bit].material : w->material[id]; };
    auto wedge_in = [&](uint32_t v, uint32_t chart) -> uint32_t {
        for (const auto& [c, x] : chart_wedges[v])
            if (c == chart) return x;
        return UINT32_MAX;
    };

    std::vector<std::vector<uint32_t>> vertex_tris(nv);
    for (size_t t = 0; t < n; ++t)
        for (int c = 0; c < 3; ++c) vertex_tris[tri[3 * t + c]].push_back(static_cast<uint32_t>(t));

    auto face_normal = [&](size_t t) {  // not normalized: length is twice the area
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
    // posed: local positions and quadrics per pose, pose k's at index k * nv.
    const size_t pose_count = poses ? poses->size() : 0;
    std::vector<vec3> posed(pose_count * nv);
    std::vector<quadric> posed_q(pose_count * nv);
    for (size_t k = 0; k < pose_count; ++k) {
        vec3* pp = &posed[k * nv];
        for (size_t v = 0; v < nv; ++v) pp[v] = (*poses)[k][verts[v]];
        for (size_t t = 0; t < n; ++t) {
            const vec3 a = pp[tri[3 * t]], b = pp[tri[3 * t + 1]], c = pp[tri[3 * t + 2]];
            const vec3 fn = cross(b - a, c - a);
            const double len = length(fn);
            if (len <= 0) continue;
            const double nx = fn.x / len, ny = fn.y / len, nz = fn.z / len;
            const quadric plane = quadric::plane(nx, ny, nz, -(nx * a.x + ny * a.y + nz * a.z), 0.5 * len);
            for (int corner = 0; corner < 3; ++corner) posed_q[k * nv + tri[3 * t + corner]] += plane;
        }
    }

    // edges, once each, with use counts. an edge used an odd number of times is a border (once,
    // or three times where vertex clustering kept duplicates): a plane through it,
    // perpendicular to its triangle, keeps collapses from pulling the border in.
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
        if (e.uses % 2 == 0) continue;
        border[e.a] = border[e.b] = 1;
        const vec3 fn = normalize(face_normal(e.tri));
        const vec3 along = pos[e.b] - pos[e.a];
        const vec3 n_border = normalize(cross(along, fn));
        if (dot(n_border, n_border) == 0) continue;
        const float d = -dot(n_border, pos[e.a]);
        // weighted as a triangle with the edge's length on each side.
        const quadric plane = quadric::plane(n_border.x, n_border.y, n_border.z, d, dot(along, along));
        q[e.a] += plane;
        q[e.b] += plane;
    }

    std::vector<uint32_t> version(nv, 0);
    std::vector<uint8_t> tri_alive(n, 1);
    std::priority_queue<collapse, std::vector<collapse>, std::greater<collapse>> heap;

    // the rest pose's mean squared distance, or posed, the worst pose's.
    auto cost_of = [&](uint32_t from, uint32_t to) {
        quadric sum = q[from];
        sum += q[to];
        double cost = sum.error(pos[to]) / std::max(sum.weight, 1e-30);
        if (!nrm.empty()) {
            const vec3 dn = nrm[to] - nrm[from], dp = pos[to] - pos[from];
            cost += normal_weight2 * double(dot(dn, dn)) * double(dot(dp, dp));
        }
        for (size_t k = 0; k < pose_count; ++k) {
            quadric ps = posed_q[k * nv + from];
            ps += posed_q[k * nv + to];
            cost = std::max(cost, ps.error(posed[k * nv + to]) / std::max(ps.weight, 1e-30));
        }
        return cost;
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

        // triangles on the edge, and the link condition: vertices next to both
        // ends must be the far corners of those triangles, else the collapse
        // pinches.
        ++mark_id;
        uint32_t shared = 0;
        for (uint32_t t : vertex_tris[from]) {
            if (!tri_alive[t]) continue;
            bool has_to = false;
            for (int k = 0; k < 3; ++k) has_to |= tri[3 * t + k] == to;
            shared += has_to;
            for (int k = 0; k < 3; ++k) mark[tri[3 * t + k]] = mark_id;
        }
        if (shared == 0) continue;  // edge gone
        // border vertices slide along the border onto the next border vertex only: along an
        // edge the remaining triangles use an odd number of times.
        if (border[from] && shared % 2 == 0) continue;
        // no collapse may delete a piece of surface: no triangle left at `to`
        // means an island vanished.
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
                    mark[v] = mark_id;  // count each once
                    ++common;
                }
            }
        }
        if (common != shared) continue;

        // moving from onto to must not flip or flatten a remaining triangle.
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
        // every triangle staying at `from` needs `to` in its texture chart.
        if (w)
            for (uint32_t t : vertex_tris[from]) {
                if (!tri_alive[t]) continue;
                const uint32_t* v = &tri[3 * t];
                if (v[0] == to || v[1] == to || v[2] == to) continue;
                const int k = v[0] == from ? 0 : v[1] == from ? 1 : 2;
                if (!w->relax && wedge_in(to, chart_of(corner[3 * t + k])) == UINT32_MAX) { ok = false; break; }
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
                    if (v[k] == from) {
                        v[k] = to;
                        if (w) {
                            const uint32_t old = corner[3 * t + k], chart = chart_of(old);
                            uint32_t x = wedge_in(to, chart);
                            if (x == UINT32_MAX) {  // relaxed: a new wedge of this chart at `to`
                                x = new_wedge_bit | uint32_t(out.new_wedges.size());
                                out.new_wedges.push_back({uv_of(old), verts[to], chart, material_of(old)});
                                chart_wedges[to].push_back({chart, x});
                            }
                            corner[3 * t + k] = x;
                        }
                    }
                vertex_tris[to].push_back(t);
            }
        }
        vertex_tris[from].clear();
        q[to] += q[from];
        for (size_t k = 0; k < pose_count; ++k) posed_q[k * nv + to] += posed_q[k * nv + from];
        ++version[from];
        ++version[to];

        // edges at to changed cost: requeue.
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
            for (int k = 0; k < 3; ++k) {
                out.indices.push_back(verts[tri[3 * t + k]]);
                if (w) out.corners.push_back(corner[3 * t + k]);
            }
    out.error = static_cast<float>(std::sqrt(worst));
    return out;
}

simplify_result cluster_vertices(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices,
                                 size_t target_triangles, const wedges* w, const std::vector<uint8_t>* locked) {
    auto is_locked = [&](uint32_t v) { return locked && (*locked)[v]; };
    // each vertex's wedges by chart.
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> chart_wedges;
    if (w)
        for (size_t i = 0; i < indices.size(); ++i) {
            auto& list = chart_wedges[indices[i]];
            const uint32_t chart = w->chart[w->corners[i]];
            bool have = false;
            for (const auto& [c, x] : list) have |= c == chart;
            if (!have) list.push_back({chart, w->corners[i]});
        }
    auto wedge_for = [&](uint32_t v, uint32_t from_wedge) {
        const auto& list = chart_wedges[v];
        for (const auto& [c, x] : list)
            if (c == w->chart[from_wedge]) return x;
        return list[0].second;
    };
    // open edges (used an odd number of times: duplicates are kept) give border vertices.
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (size_t t = 0; t < indices.size(); t += 3)
        for (int c = 0; c < 3; ++c) {
            const uint32_t a = indices[t + c], b = indices[t + (c + 1) % 3];
            edges.push_back({std::min(a, b), std::max(a, b)});
        }
    std::sort(edges.begin(), edges.end());
    std::unordered_map<uint32_t, bool> border;  // every used vertex, and whether it is on the border
    for (uint32_t v : indices) border[v] = false;
    for (size_t k = 0; k < edges.size();) {
        size_t j = k;
        while (j < edges.size() && edges[j] == edges[k]) ++j;
        if ((j - k) % 2 == 1) border[edges[k].first] = border[edges[k].second] = true;  // odd: a rim, duplicates or not
        k = j;
    }
    vec3 lo = positions[indices[0]], hi = lo;
    for (const auto& [v, b] : border) {
        lo = min(lo, positions[v]);
        hi = max(hi, positions[v]);
    }
    const vec3 size = hi - lo;
    const float extent = std::max(size.x, std::max(size.y, size.z));

    simplify_result r;
    for (int cells = 64; cells >= 1; cells = cells * 3 / 4) {
        const float step = extent / cells * 1.0001f + 1e-12f;
        auto cell_of = [&](uint32_t v) {
            const vec3 q = (positions[v] - lo) * (1 / step);
            return uint64_t(q.x) | uint64_t(q.y) << 21 | uint64_t(q.z) << 42;
        };
        // each cell's vertex: a border vertex if any, else the one nearest the
        // cell's mean.
        // a cell with locked vertices: the others snap onto its lowest locked one, and every
        // locked vertex stays.
        struct cell { vec3 sum{0, 0, 0}; uint32_t count = 0; bool border = false; uint32_t locked = UINT32_MAX; };
        std::unordered_map<uint64_t, cell> grid;
        for (const auto& [v, b] : border) {
            cell& c = grid[cell_of(v)];
            c.sum += positions[v];
            ++c.count;
            c.border |= b;
            if (is_locked(v)) c.locked = std::min(c.locked, v);
        }
        std::unordered_map<uint64_t, std::pair<uint32_t, float>> pick;
        for (const auto& [v, b] : border) {
            const uint64_t k = cell_of(v);
            const cell& c = grid[k];
            if (c.locked != UINT32_MAX) {
                pick[k] = {c.locked, 0};
                continue;
            }
            if (c.border && !b) continue;
            const float d = length(positions[v] - c.sum * (1.0f / float(c.count)));
            auto it = pick.find(k);
            if (it == pick.end() || d < it->second.second || (d == it->second.second && v < it->second.first)) pick[k] = {v, d};
        }
        // a hole's rim vertex in a cell with locked ones stays too: snapped onto a locked vertex it
        // would drag the rim across the outline.
        auto target = [&](uint32_t v) {
            if (is_locked(v)) return v;
            const uint64_t k = cell_of(v);
            if (border[v] && grid[k].locked != UINT32_MAX) return v;
            return pick[k].first;
        };
        std::vector<uint32_t> out, corners;
        for (size_t t = 0; t < indices.size(); t += 3) {
            const uint32_t a = target(indices[t]), b = target(indices[t + 1]), c = target(indices[t + 2]);
            if (a == b || b == c || a == c) continue;
            out.insert(out.end(), {a, b, c});
            if (w)
                corners.insert(corners.end(), {wedge_for(a, w->corners[t]), wedge_for(b, w->corners[t + 1]),
                                               wedge_for(c, w->corners[t + 2])});
        }
        r.indices = std::move(out);
        r.corners = std::move(corners);
        if (r.indices.size() / 3 <= target_triangles || cells == 1) {
            for (const auto& [v, b] : border) r.snapped[v] = target(v);
            break;
        }
    }
    return r;
}

void grow_to_area(const std::vector<vec3>& positions, const std::vector<uint32_t>& input, simplify_result& r,
                  const std::vector<uint8_t>& locked, const wedges* w, float max_scale) {
    if (r.indices.empty()) return;
    auto area = [&](const uint32_t* t) {
        return 0.5f * length(cross(positions[t[1]] - positions[t[0]], positions[t[2]] - positions[t[0]]));
    };
    // pieces: the result's triangles joined through shared vertices.
    std::unordered_map<uint32_t, uint32_t> parent;
    std::function<uint32_t(uint32_t)> find = [&](uint32_t v) {
        uint32_t root = v;
        while (parent[root] != root) root = parent[root];
        while (parent[v] != root) { const uint32_t next = parent[v]; parent[v] = root; v = next; }
        return root;
    };
    for (uint32_t v : r.indices) parent.emplace(v, v);
    for (size_t t = 0; t < r.indices.size(); t += 3) {
        const uint32_t a = find(r.indices[t]);
        for (int k = 1; k < 3; ++k) {
            const uint32_t b = find(r.indices[t + k]);
            if (a != b) parent[b] = a;
        }
    }
    struct piece { double have = 0, want = 0; vec3 centre{0, 0, 0}; };
    std::unordered_map<uint32_t, piece> pieces;
    for (size_t t = 0; t < r.indices.size(); t += 3) {
        const float a = area(&r.indices[t]);
        piece& p = pieces[find(r.indices[t])];
        p.have += a;
        p.centre += (positions[r.indices[t]] + positions[r.indices[t + 1]] + positions[r.indices[t + 2]]) * (a / 3);
    }
    // each input triangle's area to the piece its first corner snapped into, else the piece
    // with the nearest vertex (a needle that collapsed to a point).
    std::vector<uint32_t> kept;
    for (const auto& [v, root] : parent) kept.push_back(v);
    std::sort(kept.begin(), kept.end());
    for (size_t t = 0; t < input.size(); t += 3) {
        const float a = area(&input[t]);
        auto it = r.snapped.find(input[t]);
        uint32_t to = it == r.snapped.end() ? input[t] : it->second;
        if (!parent.count(to)) {
            float best = INFINITY;
            for (uint32_t v : kept) {
                const vec3 off = positions[v] - positions[input[t]];
                const float d = dot(off, off);
                if (d < best) { best = d; to = v; }
            }
        }
        pieces[find(to)].want += a;
    }
    // grow each piece short of its area about its area's centre; locked vertices stay.
    std::unordered_map<uint32_t, uint32_t> moved;  // vertex: new_vertex_bit | k
    for (auto& [root, p] : pieces) {
        if (p.have <= 0 || p.want <= p.have * 1.02) continue;
        const float scale = std::min(float(std::sqrt(p.want / p.have)), max_scale);
        const vec3 centre = p.centre * float(1 / p.have);
        for (const auto& [v, rv] : parent) {
            if (locked[v] || find(v) != root) continue;
            moved[v] = new_vertex_bit | uint32_t(r.new_positions.size());
            r.new_positions.push_back(centre + (positions[v] - centre) * scale);
            r.new_from.push_back(v);
        }
    }
    if (moved.empty()) return;
    // corners follow: untextured, a corner is its vertex; textured, a moved vertex gets a wedge
    // per wedge it had, with the same coordinates.
    std::unordered_map<uint64_t, uint32_t> made;
    for (size_t i = 0; i < r.indices.size(); ++i) {
        auto it = moved.find(r.indices[i]);
        if (it == moved.end()) continue;
        if (!w || w->uv.empty()) {
            if (!r.corners.empty()) r.corners[i] = it->second;
        } else {
            const uint32_t old = r.corners[i];
            const uint64_t key = uint64_t(it->second) << 32 | old;
            auto [m, added] = made.emplace(key, new_wedge_bit | uint32_t(r.new_wedges.size()));
            if (added) {
                const bool fresh = old & new_wedge_bit;
                const new_wedge& from = fresh ? r.new_wedges[old & ~new_wedge_bit] : new_wedge{};
                r.new_wedges.push_back({fresh ? from.uv : w->uv[old], it->second, fresh ? from.chart : w->chart[old],
                                        fresh ? from.material : (w->material.empty() ? uint8_t(0) : w->material[old])});
            }
            r.corners[i] = m->second;
        }
        r.indices[i] = it->second;
    }
}
