#include "dag.hpp"

#include "cluster.hpp"
#include "deviation.hpp"
#include "parallel.hpp"
#include "simplify.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
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
            const uint32_t w = static_cast<uint32_t>(j - k);
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

lod_mesh build_lod(const mesh& m, bool verbose) {
    const auto start = std::chrono::steady_clock::now();
    lod_mesh out;
    out.positions = m.positions;
    out.normals = vertex_normals(m);

    auto make_clusters = [&](const std::vector<uint32_t>& indices, std::vector<lod_cluster>& dst) {
        for (const auto& tris : clusterize(out.positions, indices)) {
            lod_cluster c;
            c.indices.reserve(3 * tris.size());
            for (uint32_t t : tris) c.indices.insert(c.indices.end(), &indices[3 * t], &indices[3 * t + 3]);
            dst.push_back(std::move(c));
        }
    };

    make_clusters(m.indices, out.clusters);
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

        // a vertex is locked if two groups use it.
        std::fill(owner.begin(), owner.end(), UINT32_MAX);
        std::fill(locked.begin(), locked.end(), 0);
        for (uint32_t g = 0; g < groups.size(); ++g)
            for (uint32_t c : groups[g])
                for (uint32_t v : out.clusters[c].indices) {
                    if (owner[v] == UINT32_MAX) owner[v] = g;
                    else if (owner[v] != g) locked[v] = 1;
                }

        std::vector<std::vector<lod_cluster>> made(groups.size());
        std::vector<uint8_t> stuck(groups.size(), 0);
        parallel_for(groups.size(), [&](size_t g) {
            std::vector<uint32_t> merged;
            float child_error = 0;
            sphere bounds = out.clusters[groups[g][0]].lod_bounds;
            for (uint32_t c : groups[g]) {
                const lod_cluster& cl = out.clusters[c];
                merged.insert(merged.end(), cl.indices.begin(), cl.indices.end());
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
            simplify_result s = simplify(out.positions, merged, locked, tris / 2);
            if ((s.indices.empty() || s.indices.size() / 3 > stuck_ratio * tris) && last)
                s = cluster_vertices(out.positions, merged, std::max<size_t>(tris / 2, 1));
            if (s.indices.empty() || s.indices.size() / 3 > stuck_ratio * tris) {
                stuck[g] = 1;
                return;
            }
            // choosing between the two simplifications by sampled distance (cheap, and the
            // choice need not be proven), then proving the chosen one's: refined until within
            // 10% of the worst found or under a millionth of the group (where the levels
            // coincide, flat parts, a bound cannot reach 0).
            float sampled = sampled_deviation(out.positions, merged, s.indices);
            if (sampled > outlier_ratio * s.error) {
                simplify_result gentle = simplify(out.positions, merged, locked, tris * 3 / 4);
                if (!gentle.indices.empty() && gentle.indices.size() / 3 <= stuck_ratio * tris) {
                    const float gentle_sampled = sampled_deviation(out.positions, merged, gentle.indices);
                    if (gentle_sampled < sampled) {
                        s = std::move(gentle);
                        sampled = gentle_sampled;
                    }
                }
            }
            const float measured = mesh_deviation(out.positions, merged, s.indices, 0.1f, bounds.radius * 1e-6f);
            const float error = child_error + std::max(s.error, measured);
            make_clusters(s.indices, made[g]);
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
            for (lod_cluster& c : made[g]) {
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
