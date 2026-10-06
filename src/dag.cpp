#include "dag.hpp"

#include "cluster.hpp"
#include "deviation.hpp"
#include "parallel.hpp"
#include "simplify.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <numeric>

namespace {

constexpr uint32_t group_target = 8;  // clusters per group
// a group keeping more than this share of its triangles is left as is: its
// clusters become roots.
constexpr float stuck_ratio = 0.85f;
constexpr uint32_t max_levels = 48;
// a simplification measuring over this many times its quadric estimate is
// retried more gently (see build_lod).
constexpr float outlier_ratio = 4.0f;
constexpr size_t min_group_triangles = 64;
// clustered foliage grows back the area its thin pieces lost (grow_to_area()), at most this
// much across.
constexpr bool preserve_area = true;
constexpr float max_growth = 3.0f;
constexpr float foliage_rims = 0.25f;  // share of a group's edges on rims that makes it foliage

// the total area of triangles.
float area(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices) {
    double sum = 0;
    for (size_t t = 0; t < indices.size(); t += 3)
        sum += 0.5 * length(cross(positions[indices[t + 1]] - positions[indices[t]], positions[indices[t + 2]] - positions[indices[t]]));
    return float(sum);
}

// the share of edges used an odd number of times (rims).
float rim_share(const std::vector<uint32_t>& indices) {
    std::vector<uint64_t> e;
    e.reserve(indices.size());
    for (size_t t = 0; t < indices.size(); t += 3)
        for (int c = 0; c < 3; ++c) {
            const uint32_t a = indices[t + c], b = indices[t + (c + 1) % 3];
            e.push_back(uint64_t(std::min(a, b)) << 32 | std::max(a, b));
        }
    std::sort(e.begin(), e.end());
    size_t edges = 0, rims = 0;
    for (size_t k = 0; k < e.size();) {
        size_t j = k;
        while (j < e.size() && e[j] == e[k]) ++j;
        ++edges;
        rims += (j - k) % 2;
        k = j;
    }
    return edges ? float(rims) / float(edges) : 0;
}
// go on until one cluster of at most this many triangles remains: every
// instance draws at least its root.
constexpr size_t root_triangles = 32;

// groups clusters (numbers into all) into sets of about group_target sharing
// the most edges.
//
// greedy: seeds in spatial order; a group takes the unassigned neighbour it
// shares most edges with until full or out of neighbours. small groups merge
// into their best-connected neighbour.
std::vector<std::vector<uint32_t>> partition(const std::vector<lod_cluster>& all, const std::vector<uint32_t>& level) {
    const size_t n = level.size();

    // each cluster's border edges, tagged with it; sorted, equal edges from two
    // clusters make them neighbours, weighted by edges shared.
    struct tagged { uint32_t a, b, cluster; };
    std::vector<std::vector<tagged>> per_cluster(n);
    parallel_for(n, [&](size_t i) {
        const auto& idx = all[level[i]].indices;
        std::vector<std::pair<uint32_t, uint32_t>> e;
        for (size_t t = 0; t < idx.size(); t += 3)
            for (int c = 0; c < 3; ++c) {
                const uint32_t a = idx[t + c], b = idx[t + (c + 1) % 3];
                e.push_back({std::min(a, b), std::max(a, b)});
            }
        std::sort(e.begin(), e.end());
        for (size_t k = 0; k < e.size();) {
            size_t j = k;
            while (j < e.size() && e[j] == e[k]) ++j;
            if (j - k == 1) per_cluster[i].push_back({e[k].first, e[k].second, static_cast<uint32_t>(i)});
            k = j;
        }
    });
    std::vector<tagged> edges;
    for (auto& v : per_cluster) edges.insert(edges.end(), v.begin(), v.end());
    per_cluster.clear();
    std::sort(edges.begin(), edges.end(), [](const tagged& x, const tagged& y) {
        return x.a != y.a ? x.a < y.a : x.b != y.b ? x.b < y.b : x.cluster < y.cluster;
    });
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> adjacency(n);  // (neighbour, shared edges)
    {
        std::vector<std::pair<uint32_t, uint32_t>> pairs;
        for (size_t k = 0; k < edges.size();) {
            size_t j = k;
            while (j < edges.size() && edges[j].a == edges[k].a && edges[j].b == edges[k].b) ++j;
            for (size_t x = k; x < j; ++x)
                for (size_t y = x + 1; y < j; ++y)
                    if (edges[x].cluster != edges[y].cluster) pairs.push_back({edges[x].cluster, edges[y].cluster});
            k = j;
        }
        edges.clear();
        std::sort(pairs.begin(), pairs.end());
        for (size_t k = 0; k < pairs.size();) {
            size_t j = k;
            while (j < pairs.size() && pairs[j] == pairs[k]) ++j;
            const uint32_t w = static_cast<uint32_t>(j - k) * 64;  // spatial links (below) weigh 1
            adjacency[pairs[k].first].push_back({pairs[k].second, w});
            adjacency[pairs[k].second].push_back({pairs[k].first, w});
            k = j;
        }
    }

    // seeds in morton order of cluster centres.
    vec3 lo = all[level[0]].bounds.center, hi = lo;
    for (uint32_t c : level) { lo = min(lo, all[c].bounds.center); hi = max(hi, all[c].bounds.center); }
    std::vector<uint64_t> code(n);
    const vec3 size = max(hi - lo, vec3(1e-30f, 1e-30f, 1e-30f));
    for (size_t i = 0; i < n; ++i) {
        const vec3 q = all[level[i]].bounds.center - lo;
        uint64_t k = 0;
        const uint32_t x = static_cast<uint32_t>(q.x / size.x * 2097151), y = static_cast<uint32_t>(q.y / size.y * 2097151),
                       z = static_cast<uint32_t>(q.z / size.z * 2097151);
        for (int bit = 20; bit >= 0; --bit) k = (k << 3) | ((x >> bit & 1) << 2) | ((y >> bit & 1) << 1) | (z >> bit & 1);
        code[i] = k;
    }
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return code[a] < code[b]; });

    // islands (needles, leaves, blades: clusters with at most one neighbour by edges) would make
    // groups of their own, too small to simplify: they link to their four nearest clusters
    // (among the 32 round them in morton order), weighing less than any shared edge.
    {
        std::vector<std::vector<uint32_t>> near(n);
        parallel_for(n, [&](size_t k) {
            const uint32_t c = order[k];
            if (adjacency[c].size() > 1) return;
            std::vector<std::pair<float, uint32_t>> cand;
            for (size_t j = k >= 16 ? k - 16 : 0; j < std::min(n, k + 17); ++j)
                if (j != k) cand.push_back({length(all[level[order[j]]].bounds.center - all[level[c]].bounds.center), order[j]});
            std::partial_sort(cand.begin(), cand.begin() + std::min<size_t>(4, cand.size()), cand.end());
            for (size_t q = 0; q < std::min<size_t>(4, cand.size()); ++q) near[c].push_back(cand[q].second);
        });
        for (uint32_t c = 0; c < n; ++c)
            for (uint32_t o : near[c]) {
                auto has = [&](uint32_t a, uint32_t b) {
                    for (auto [x, w] : adjacency[a]) if (x == b) return true;
                    return false;
                };
                if (has(c, o)) continue;
                adjacency[c].push_back({o, 1});
                adjacency[o].push_back({c, 1});
            }
    }

    std::vector<uint32_t> group_of(n, UINT32_MAX);
    std::vector<std::vector<uint32_t>> groups;
    std::vector<uint32_t> weight(n, 0), touched;
    for (uint32_t seed : order) {
        if (group_of[seed] != UINT32_MAX) continue;
        const uint32_t g = static_cast<uint32_t>(groups.size());
        groups.push_back({seed});
        group_of[seed] = g;
        touched.clear();
        vec3 center_sum = all[level[seed]].bounds.center;
        auto take = [&](uint32_t c) {
            for (auto [nb, w] : adjacency[c]) {
                if (group_of[nb] != UINT32_MAX) continue;
                if (weight[nb] == 0) touched.push_back(nb);
                weight[nb] += w;
            }
        };
        take(seed);
        while (groups[g].size() < group_target) {
            const vec3 center = center_sum * (1.0f / groups[g].size());
            uint32_t best = UINT32_MAX;
            float best_dist = INFINITY;
            for (uint32_t c : touched) {
                if (group_of[c] != UINT32_MAX) continue;
                const float d = length(all[level[c]].bounds.center - center);
                if (best == UINT32_MAX || weight[c] > weight[best] || (weight[c] == weight[best] && d < best_dist)) {
                    best = c;
                    best_dist = d;
                }
            }
            if (best == UINT32_MAX) break;
            group_of[best] = g;
            groups[g].push_back(best);
            center_sum += all[level[best]].bounds.center;
            take(best);
        }
        for (uint32_t c : touched) weight[c] = 0;
    }

    // merge small groups into their best-connected neighbour.
    for (size_t g = 0; g < groups.size(); ++g) {
        if (groups[g].empty() || groups[g].size() > group_target / 2) continue;
        std::vector<std::pair<uint32_t, uint32_t>> links;  // (group, shared edges)
        for (uint32_t c : groups[g])
            for (auto [nb, w] : adjacency[c])
                if (group_of[nb] != g) links.push_back({group_of[nb], w});
        std::sort(links.begin(), links.end());
        uint32_t best = UINT32_MAX, best_w = 0;
        for (size_t k = 0; k < links.size();) {
            size_t j = k;
            uint32_t w = 0;
            while (j < links.size() && links[j].first == links[k].first) w += links[j++].second;
            const uint32_t other = links[k].first;
            if (w > best_w && groups[other].size() + groups[g].size() <= group_target + group_target / 2) {
                best = other;
                best_w = w;
            }
            k = j;
        }
        if (best == UINT32_MAX) continue;
        for (uint32_t c : groups[g]) {
            group_of[c] = best;
            groups[best].push_back(c);
        }
        groups[g].clear();
    }

    std::vector<std::vector<uint32_t>> out;
    for (auto& g : groups) {
        if (g.empty()) continue;
        for (uint32_t& c : g) c = level[c];
        out.push_back(std::move(g));
    }
    return out;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// the mesh's wedges, split so each lies in one texture chart, and a wedge per corner. charts
// join triangles across an edge whose two ends have the same wedges on both sides. an
// untextured mesh gets a wedge per vertex, one chart.
std::vector<uint32_t> make_wedges(const mesh& m, lod_mesh& out) {
    const size_t n = m.triangle_count();
    if (!m.textured()) {
        out.wedge_vertex.resize(m.positions.size());
        std::iota(out.wedge_vertex.begin(), out.wedge_vertex.end(), 0u);
        out.wedge_chart.assign(m.positions.size(), 0);
        return m.indices;
    }
    // union find over triangles.
    std::vector<uint32_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0u);
    auto find = [&](uint32_t x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    };
    struct side { uint64_t edge; uint32_t tri, wa, wb; };
    std::vector<side> sides;
    sides.reserve(3 * n);
    for (uint32_t t = 0; t < n; ++t)
        for (int c = 0; c < 3; ++c) {
            uint32_t a = m.indices[3 * t + c], b = m.indices[3 * t + (c + 1) % 3];
            uint32_t wa = m.corners[3 * t + c], wb = m.corners[3 * t + (c + 1) % 3];
            if (a > b) { std::swap(a, b); std::swap(wa, wb); }
            sides.push_back({uint64_t(a) << 32 | b, t, wa, wb});
        }
    std::sort(sides.begin(), sides.end(), [](const side& x, const side& y) { return x.edge < y.edge; });
    for (size_t i = 0; i < sides.size();) {
        size_t j = i;
        while (j < sides.size() && sides[j].edge == sides[i].edge) ++j;
        for (size_t k = i + 1; k < j; ++k)
            if (sides[k].wa == sides[i].wa && sides[k].wb == sides[i].wb) parent[find(sides[k].tri)] = find(sides[i].tri);
        i = j;
    }
    // a wedge per (wedge, chart): a wedge two charts share splits.
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> split;
    std::vector<uint32_t> corners(m.corners.size());
    for (uint32_t t = 0; t < n; ++t) {
        const uint32_t chart = find(t);
        for (int c = 0; c < 3; ++c) {
            const uint32_t w = m.corners[3 * t + c];
            auto [it, added] = split.emplace(std::make_pair(w, chart), uint32_t(out.wedge_uvs.size()));
            if (added) {
                out.wedge_uvs.push_back(m.wedge_uvs[w]);
                out.wedge_material.push_back(m.wedge_material[w]);
                out.wedge_vertex.push_back(m.wedge_vertex[w]);
                out.wedge_chart.push_back(chart);
            }
            corners[3 * t + c] = it->second;
        }
    }
    out.materials = m.materials;
    return corners;
}

}  // namespace

void cluster_bounds(const std::vector<vec3>& positions, lod_cluster& c) {
    const auto& idx = c.indices;
    c.bounds = bounding_sphere(idx.size(), [&](size_t i) { return positions[idx[i]]; });

    vec3 axis(0, 0, 0);
    std::vector<vec3> normals;
    for (size_t t = 0; t < idx.size(); t += 3) {
        const vec3 fn = cross(positions[idx[t + 1]] - positions[idx[t]], positions[idx[t + 2]] - positions[idx[t]]);
        if (dot(fn, fn) > 0) normals.push_back(normalize(fn));
    }
    for (const vec3& fn : normals) axis += fn;
    c.cone_axis = {0, 0, 1};
    c.cone_cutoff = 1;
    if (dot(axis, axis) > 1e-12f) {
        axis = normalize(axis);
        float min_dot = 1;
        for (const vec3& fn : normals) min_dot = std::min(min_dot, dot(fn, axis));
        c.cone_axis = axis;
        // normals spread past about 84 degrees cull too little to pay.
        if (min_dot > 0.1f) c.cone_cutoff = std::sqrt(1 - min_dot * min_dot);
    }
}

lod_mesh build_lod(const mesh& m, bool verbose, const std::vector<std::vector<float>>& poses) {
    const auto start = std::chrono::steady_clock::now();
    lod_mesh out;
    out.positions = m.positions;
    out.normals = vertex_normals(m);
    out.skin_joints = m.skin_joints;
    out.skin_weights = m.skin_weights;
    std::vector<uint32_t> corners = make_wedges(m, out);
    // skinned: the mesh posed. every pose measures errors; every fourth (the rest pose first)
    // also prices collapses (simplify()).
    auto pose_mesh = [&](const std::vector<float>& rows) {
        std::vector<vec3> at(out.positions.size());
        for (size_t v = 0; v < at.size(); ++v) {
            const vec3 p = out.positions[v];
            vec3 q(0, 0, 0);
            for (int k = 0; k < 4; ++k) {
                const float w = m.skin_weights[4 * v + k] / 255.0f;
                if (w == 0) continue;
                const float* r = &rows[12 * size_t(m.skin_joints[4 * v + k])];
                q += vec3(r[0] * p.x + r[1] * p.y + r[2] * p.z + r[3], r[4] * p.x + r[5] * p.y + r[6] * p.z + r[7],
                          r[8] * p.x + r[9] * p.y + r[10] * p.z + r[11]) * w;
            }
            at[v] = q;
        }
        return at;
    };
    std::vector<std::vector<vec3>> pricing;
    if (m.skinned())
        for (size_t k = 0; k < poses.size(); k += 4) pricing.push_back(pose_mesh(poses[k]));
    const std::vector<std::vector<vec3>>* priced = pricing.empty() ? nullptr : &pricing;

    auto make_clusters = [&](const std::vector<uint32_t>& indices, const std::vector<uint32_t>& corner_list,
                             std::vector<lod_cluster>& dst) {
        for (const auto& tris : clusterize(out.positions, indices, out.wedge_uvs.empty() ? nullptr : &corner_list)) {
            lod_cluster c;
            c.indices.reserve(3 * tris.size());
            c.corners.reserve(3 * tris.size());
            for (uint32_t t : tris) {
                c.indices.insert(c.indices.end(), &indices[3 * t], &indices[3 * t + 3]);
                c.corners.insert(c.corners.end(), &corner_list[3 * t], &corner_list[3 * t + 3]);
            }
            dst.push_back(std::move(c));
        }
    };

    make_clusters(m.indices, corners, out.clusters);
    parallel_for(out.clusters.size(), [&](size_t i) {
        cluster_bounds(out.positions, out.clusters[i]);
        out.clusters[i].lod_bounds = out.clusters[i].bounds;
    });

    std::vector<uint32_t> level(out.clusters.size());
    std::iota(level.begin(), level.end(), 0);
    std::vector<uint32_t> owner(out.positions.size());
    std::vector<uint8_t> locked(out.positions.size());
    uint32_t group_base = 0;

    for (uint32_t depth = 0;; ++depth) {
        lod_level_stats stats;
        stats.clusters = level.size();
        for (uint32_t c : level) {
            stats.triangles += out.clusters[c].indices.size() / 3;
            stats.max_error = std::max(stats.max_error, out.clusters[c].lod_error);
        }
        if ((level.size() <= 1 && stats.triangles <= root_triangles) || depth + 1 >= max_levels) {
            out.levels.push_back(stats);
            if (verbose)
                std::printf("  level %2u: %8zu clusters %10zu triangles  (root)\n", depth, stats.clusters, stats.triangles);
            break;
        }

        const auto groups = partition(out.clusters, level);
        stats.groups = groups.size();

        // a vertex is locked if two groups use it. (foliage grows new vertices: sized afresh.)
        owner.assign(out.positions.size(), UINT32_MAX);
        locked.assign(out.positions.size(), 0);
        for (uint32_t g = 0; g < groups.size(); ++g)
            for (uint32_t c : groups[g])
                for (uint32_t v : out.clusters[c].indices) {
                    if (owner[v] == UINT32_MAX) owner[v] = g;
                    else if (owner[v] != g) locked[v] = 1;
                }

        std::vector<std::vector<lod_cluster>> made(groups.size());
        std::vector<std::vector<new_wedge>> made_wedges(groups.size());
        std::vector<uint8_t> stuck(groups.size(), 0);
        std::vector<std::pair<std::vector<vec3>, std::vector<uint32_t>>> grown(groups.size());  // new vertices, and whose
        parallel_for(groups.size(), [&](size_t g) {
            std::vector<uint32_t> merged, merged_corners;
            float child_error = 0;
            sphere bounds = out.clusters[groups[g][0]].lod_bounds;
            for (uint32_t c : groups[g]) {
                const lod_cluster& cl = out.clusters[c];
                merged.insert(merged.end(), cl.indices.begin(), cl.indices.end());
                merged_corners.insert(merged_corners.end(), cl.corners.begin(), cl.corners.end());
                child_error = std::max(child_error, cl.lod_error);
                bounds = merge(bounds, cl.lod_bounds);
            }
            const size_t tris = merged.size() / 3;
            // the last group, the whole model: nothing borders it, so when
            // collapses stall (thin parts, hole rims) vertex clustering takes
            // over.
            const bool last = groups.size() == 1;
            // errors add: child_error plus how far this level is from the last,
            // measured, not the quadric estimate (a mean, typically under half
            // the worst spot); the larger is kept.
            //
            // groups far worse than their estimate (a thin part folded flat, a
            // rim pulled across) are retried more gently and the better kept.
            // the measured error is recorded either way: a bad spot costs
            // detail, not correctness. groups of a few triangles are scan
            // islands; halving one reshapes it (the dragon's worst level 0
            // groups, 6 to 0.4 percent of its size). they stay as roots at full
            // detail.
            if (tris < min_group_triangles && !last) {
                stuck[g] = 1;
                return;
            }
            wedges w{merged_corners, out.wedge_chart, out.wedge_vertex, out.wedge_uvs, out.wedge_material};
            simplify_result s = simplify(out.positions, merged, locked, tris / 2, &w, priced);
            // seams of many small texture charts can pin a group: let them move.
            if ((s.indices.empty() || s.indices.size() / 3 > stuck_ratio * tris) && !out.wedge_uvs.empty()) {
                w.relax = true;
                s = simplify(out.positions, merged, locked, tris / 2, &w, priced);
            }
            // still stuck: islands (needles, leaves, blades) no collapse may remove, or thin parts.
            // vertex clustering merges them, the group's outline (locked vertices) kept in place
            // so its neighbours still meet it; the last group has no outline.
            bool clustered = false;
            if (s.indices.empty() || s.indices.size() / 3 > stuck_ratio * tris) {
                s = cluster_vertices(out.positions, merged, std::max<size_t>(tris / 2, 1), &w, &locked);
                clustered = true;
            }
            if (s.indices.empty() || s.indices.size() / 3 > stuck_ratio * tris) {
                stuck[g] = 1;
                return;
            }
            // clustered foliage (a group mostly of open rims: needles, blades, cards) loses the
            // pieces thinner than a cell: what's left grows to keep their area. closed surfaces
            // (scans) are left as they are. (growing after edge collapses too compounded level on
            // level until groups stuck: the pine never reached a root.)
            const bool foliage = rim_share(merged) > foliage_rims;
            if (clustered && preserve_area && foliage) {
                grow_to_area(out.positions, merged, s, locked, &w, max_growth);
                if (!s.new_positions.empty()) {
                    // the group on its own vertices, the new ones after: errors and clusters from
                    // these, indices back to the shared numbering (new ones still marked) after.
                    std::vector<uint32_t> verts(merged);
                    for (uint32_t v : s.indices)
                        if (!(v & new_vertex_bit)) verts.push_back(v);
                    std::sort(verts.begin(), verts.end());
                    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
                    std::vector<vec3> at(verts.size());
                    for (size_t i = 0; i < verts.size(); ++i) at[i] = out.positions[verts[i]];
                    at.insert(at.end(), s.new_positions.begin(), s.new_positions.end());
                    auto local = [&](uint32_t v) {
                        return v & new_vertex_bit ? uint32_t(verts.size()) + (v & ~new_vertex_bit)
                                                  : uint32_t(std::lower_bound(verts.begin(), verts.end(), v) - verts.begin());
                    };
                    auto shared = [&](uint32_t l) { return l < verts.size() ? verts[l] : new_vertex_bit | uint32_t(l - verts.size()); };
                    std::vector<uint32_t> local_merged(merged), local_simplified(s.indices);
                    for (uint32_t& v : local_merged) v = local(v);
                    for (uint32_t& v : local_simplified) v = local(v);
                    const float measured = mesh_deviation(at, local_merged, local_simplified, 0.1f, bounds.radius * 1e-6f);
                    const float lost = foliage ? std::sqrt(std::max(0.0f, area(at, local_merged) - area(at, local_simplified))) : 0.0f;
                    const float error = child_error + std::max({s.error, measured, lost});
                    for (const auto& tris_of : clusterize(at, local_simplified, out.wedge_uvs.empty() ? nullptr : &s.corners)) {
                        lod_cluster c;
                        for (uint32_t t : tris_of) {
                            c.indices.insert(c.indices.end(), &local_simplified[3 * t], &local_simplified[3 * t + 3]);
                            c.corners.insert(c.corners.end(), &s.corners[3 * t], &s.corners[3 * t + 3]);
                        }
                        cluster_bounds(at, c);
                        for (uint32_t& v : c.indices) v = shared(v);
                        c.lod_bounds = bounds;
                        c.lod_error = error;
                        c.level = depth + 1;
                        made[g].push_back(std::move(c));
                    }
                    made_wedges[g] = std::move(s.new_wedges);
                    grown[g] = {std::move(s.new_positions), std::move(s.new_from)};
                    return;
                }
            }
            // choosing between the two simplifications by sampled distance (cheap, and the
            // choice need not be proven), then proving the chosen one's: refined until within
            // 10% of the worst found or under a millionth of the group (where the levels
            // coincide, flat parts, a bound cannot reach 0).
            float sampled = sampled_deviation(out.positions, merged, s.indices);
            if (sampled > outlier_ratio * s.error) {
                simplify_result gentle = simplify(out.positions, merged, locked, tris * 3 / 4, &w, priced);
                if (!gentle.indices.empty() && gentle.indices.size() / 3 <= stuck_ratio * tris) {
                    const float gentle_sampled = sampled_deviation(out.positions, merged, gentle.indices);
                    if (gentle_sampled < sampled) {
                        s = std::move(gentle);
                        sampled = gentle_sampled;
                    }
                }
            }
            const float measured = mesh_deviation(out.positions, merged, s.indices, 0.1f, bounds.radius * 1e-6f);
            // skinned: the same two surfaces in every pose.
            float posed = 0;
            if (!poses.empty() && m.skinned()) {
                std::vector<uint32_t> local_merged(merged), local_simplified(s.indices), verts(merged);
                std::sort(verts.begin(), verts.end());
                verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
                auto local = [&](uint32_t v) { return uint32_t(std::lower_bound(verts.begin(), verts.end(), v) - verts.begin()); };
                for (uint32_t& v : local_merged) v = local(v);
                for (uint32_t& v : local_simplified) v = local(v);
                std::vector<vec3> at(verts.size());
                for (const std::vector<float>& rows : poses) {
                    for (size_t i = 0; i < verts.size(); ++i) {
                        const vec3 p = out.positions[verts[i]];
                        vec3 q(0, 0, 0);
                        for (int k = 0; k < 4; ++k) {
                            const float w = m.skin_weights[4 * size_t(verts[i]) + k] / 255.0f;
                            if (w == 0) continue;
                            const float* r = &rows[12 * size_t(m.skin_joints[4 * size_t(verts[i]) + k])];
                            q += vec3(r[0] * p.x + r[1] * p.y + r[2] * p.z + r[3], r[4] * p.x + r[5] * p.y + r[6] * p.z + r[7],
                                      r[8] * p.x + r[9] * p.y + r[10] * p.z + r[11]) * w;
                        }
                        at[i] = q;
                    }
                    posed = std::max(posed, sampled_deviation(at, local_merged, local_simplified));
                }
            }
            // foliage: the error is also the side of the square of area it lost, so a cut drawn at
            // t pixels loses at most about t * t pixels of coverage per group. a hausdorff bound
            // alone lets a crown lose half its needles: each removed one lies near a kept one.
            const float lost = foliage ? std::sqrt(std::max(0.0f, area(out.positions, merged) - area(out.positions, s.indices))) : 0.0f;
            const float error = child_error + std::max({s.error, measured, posed, lost});
            make_clusters(s.indices, s.corners, made[g]);
            made_wedges[g] = std::move(s.new_wedges);
            for (lod_cluster& c : made[g]) {
                cluster_bounds(out.positions, c);
                c.lod_bounds = bounds;
                c.lod_error = error;
                c.level = depth + 1;
            }
        });

        // a stuck group's clusters go to the next level as they are, grouped
        // with other neighbours, locking other vertices; they usually simplify
        // there. their parent is whichever group does.
        std::vector<uint32_t> next;
        for (uint32_t g = 0; g < groups.size(); ++g) {
            if (stuck[g]) {
                ++stats.stuck_groups;
                next.insert(next.end(), groups[g].begin(), groups[g].end());
                continue;
            }
            const uint32_t group_id = group_base + g;
            for (uint32_t c : groups[g]) {
                out.clusters[c].parent_bounds = made[g][0].lod_bounds;
                out.clusters[c].parent_error = made[g][0].lod_error;
                out.clusters[c].group = group_id;
            }
            // vertices grown foliage moved, numbered after the rest, each copying its source's
            // normal and skin (untextured, its wedge is itself).
            const uint32_t vertex_base = static_cast<uint32_t>(out.positions.size());
            const auto& [new_positions, new_from] = grown[g];
            if (!new_positions.empty() && out.vertex_source.empty()) {
                out.vertex_source.resize(vertex_base);
                std::iota(out.vertex_source.begin(), out.vertex_source.end(), 0u);
            }
            for (size_t k = 0; k < new_positions.size(); ++k) {
                out.vertex_source.push_back(new_from[k]);
                out.positions.push_back(new_positions[k]);
                out.normals.push_back(out.normals[new_from[k]]);
                if (!out.skin_weights.empty())
                    for (int j = 0; j < 4; ++j) {
                        out.skin_joints.push_back(out.skin_joints[4 * size_t(new_from[k]) + j]);
                        out.skin_weights.push_back(out.skin_weights[4 * size_t(new_from[k]) + j]);
                    }
                if (out.wedge_uvs.empty()) {
                    out.wedge_vertex.push_back(vertex_base + uint32_t(k));
                    out.wedge_chart.push_back(0);
                }
            }
            auto real_vertex = [&](uint32_t v) { return v & new_vertex_bit ? vertex_base + (v & ~new_vertex_bit) : v; };
            // wedges the group's relaxed seams (or grown vertices) made, numbered after the rest.
            const uint32_t wedge_base = static_cast<uint32_t>(out.wedge_uvs.empty() ? 0 : out.wedge_uvs.size());
            for (const new_wedge& nw : made_wedges[g]) {
                out.wedge_uvs.push_back(nw.uv);
                out.wedge_material.push_back(nw.material);
                out.wedge_vertex.push_back(real_vertex(nw.vertex));
                out.wedge_chart.push_back(nw.chart);
            }
            for (lod_cluster& c : made[g]) {
                for (uint32_t& v : c.indices) v = real_vertex(v);
                if (out.wedge_uvs.empty())
                    for (uint32_t& k : c.corners) k = real_vertex(k);
                for (uint32_t& k : c.corners)
                    if (k & new_wedge_bit) k = wedge_base + (k & ~new_wedge_bit);
                c.creator = group_id;
                next.push_back(static_cast<uint32_t>(out.clusters.size()));
                out.clusters.push_back(std::move(c));
            }
        }
        out.levels.push_back(stats);
        if (verbose)
            std::printf("  level %2u: %8zu clusters %10zu triangles  %7zu groups, %zu stuck  error %.3g  (%.1fs)\n", depth,
                        stats.clusters, stats.triangles, stats.groups, stats.stuck_groups, stats.max_error,
                        seconds_since(start));
        group_base += static_cast<uint32_t>(groups.size());
        if (stats.stuck_groups == groups.size()) break;  // no progress: the rest are roots
        level = std::move(next);
    }
    return out;
}
