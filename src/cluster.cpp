#include "cluster.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr size_t leaf_triangles = 8192;

// Spreads the low 10 bits of v out to every third bit.
uint32_t spread_bits(uint32_t v) {
    v &= 1023;
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

// Clusters one piece: tris are triangle numbers into indices.
std::vector<std::vector<uint32_t>> clusterize_piece(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices,
                                                    const std::vector<uint32_t>& tris) {
    const size_t n = tris.size();
    std::vector<std::vector<uint32_t>> out;
    if (n == 0) return out;

    // Local vertex numbers, and for each vertex the triangles using it.
    std::vector<uint32_t> verts;
    verts.reserve(3 * n);
    for (uint32_t t : tris)
        for (int c = 0; c < 3; ++c) verts.push_back(indices[3 * t + c]);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    const size_t nv = verts.size();
    std::vector<uint32_t> corner(3 * n);  // Local vertex of each corner
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
            corner[3 * i + c] = static_cast<uint32_t>(std::lower_bound(verts.begin(), verts.end(), indices[3 * tris[i] + c]) - verts.begin());
    std::vector<uint32_t> adj_start(nv + 1, 0), adj(3 * n);
    for (uint32_t v : corner) ++adj_start[v + 1];
    for (size_t v = 0; v < nv; ++v) adj_start[v + 1] += adj_start[v];
    {
        std::vector<uint32_t> fill(adj_start.begin(), adj_start.end() - 1);
        for (size_t i = 0; i < 3 * n; ++i) adj[fill[corner[i]]++] = static_cast<uint32_t>(i / 3);
    }

    std::vector<vec3> centroid(n);
    vec3 lo = positions[verts[0]], hi = lo;
    double area = 0;
    for (size_t i = 0; i < n; ++i) {
        const vec3 a = positions[verts[corner[3 * i]]], b = positions[verts[corner[3 * i + 1]]], c = positions[verts[corner[3 * i + 2]]];
        centroid[i] = (a + b + c) * (1.0f / 3);
        lo = min(lo, centroid[i]);
        hi = max(hi, centroid[i]);
        area += 0.5 * length(cross(b - a, c - a));
    }
    // The radius a full cluster would have if it were a disc: the scale
    // distances are measured against.
    const float expected_radius = std::max(1e-12f, static_cast<float>(std::sqrt(area / n * cluster_max_triangles / M_PI)));

    // Seeds are taken in Morton order, so each new cluster starts next to
    // the last one and the leftover holes stay few.
    std::vector<uint32_t> order(n);
    {
        std::vector<uint32_t> code(n);
        const vec3 size = max(hi - lo, vec3(1e-30f, 1e-30f, 1e-30f));
        for (size_t i = 0; i < n; ++i) {
            const vec3 q = centroid[i] - lo;
            code[i] = spread_bits(static_cast<uint32_t>(q.x / size.x * 1023)) |
                      (spread_bits(static_cast<uint32_t>(q.y / size.y * 1023)) << 1) |
                      (spread_bits(static_cast<uint32_t>(q.z / size.z * 1023)) << 2);
        }
        for (size_t i = 0; i < n; ++i) order[i] = static_cast<uint32_t>(i);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return code[a] < code[b]; });
    }

    std::vector<char> used(n, 0);
    std::vector<uint32_t> in_cluster(nv, UINT32_MAX);  // Cluster number a vertex was last added to
    std::vector<uint32_t> candidate_stamp(n, UINT32_MAX);
    std::vector<uint32_t> candidates;
    size_t seed_cursor = 0;
    uint32_t cluster_id = 0;

    for (size_t done = 0; done < n; ++cluster_id) {
        while (used[order[seed_cursor]]) ++seed_cursor;
        std::vector<uint32_t> members;
        uint32_t vertex_count = 0;
        vec3 sum(0, 0, 0);
        candidates.clear();

        auto add = [&](uint32_t t) {
            used[t] = 1;
            ++done;
            members.push_back(t);
            sum += centroid[t];
            for (int c = 0; c < 3; ++c) {
                const uint32_t v = corner[3 * t + c];
                if (in_cluster[v] != cluster_id) {
                    in_cluster[v] = cluster_id;
                    ++vertex_count;
                }
                for (uint32_t k = adj_start[v]; k < adj_start[v + 1]; ++k) {
                    const uint32_t u = adj[k];
                    if (!used[u] && candidate_stamp[u] != cluster_id) {
                        candidate_stamp[u] = cluster_id;
                        candidates.push_back(u);
                    }
                }
            }
        };
        add(order[seed_cursor]);

        while (members.size() < cluster_max_triangles) {
            const vec3 center = sum * (1.0f / members.size());
            float best_score = INFINITY;
            size_t best = SIZE_MAX;
            for (size_t k = 0; k < candidates.size();) {
                const uint32_t t = candidates[k];
                if (used[t]) {
                    candidates[k] = candidates.back();
                    candidates.pop_back();
                    continue;
                }
                uint32_t fresh = 0;
                for (int c = 0; c < 3; ++c) fresh += in_cluster[corner[3 * t + c]] != cluster_id;
                if (vertex_count + fresh <= cluster_max_vertices) {
                    const float score = static_cast<float>(fresh) + 2 * length(centroid[t] - center) / expected_radius;
                    if (score < best_score) { best_score = score; best = k; }
                }
                ++k;
            }
            if (best == SIZE_MAX) break;  // Nothing touches the cluster that still fits
            const uint32_t t = candidates[best];
            candidates[best] = candidates.back();
            candidates.pop_back();
            add(t);
        }
        for (uint32_t& t : members) t = tris[t];
        out.push_back(std::move(members));
    }
    return out;
}

}  // namespace

std::vector<std::vector<uint32_t>> clusterize(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices) {
    const size_t n = indices.size() / 3;
    std::vector<vec3> centroid(n);
    for (size_t t = 0; t < n; ++t)
        centroid[t] = (positions[indices[3 * t]] + positions[indices[3 * t + 1]] + positions[indices[3 * t + 2]]) * (1.0f / 3);
    std::vector<uint32_t> all(n);
    for (size_t t = 0; t < n; ++t) all[t] = static_cast<uint32_t>(t);

    // Cut into pieces of at most leaf_triangles by median splits, longest axis first.
    std::vector<std::vector<uint32_t>> pieces;
    std::vector<std::vector<uint32_t>> stack;
    stack.push_back(std::move(all));
    while (!stack.empty()) {
        std::vector<uint32_t> part = std::move(stack.back());
        stack.pop_back();
        if (part.size() <= leaf_triangles) {
            pieces.push_back(std::move(part));
            continue;
        }
        vec3 lo = centroid[part[0]], hi = lo;
        for (uint32_t t : part) { lo = min(lo, centroid[t]); hi = max(hi, centroid[t]); }
        const vec3 size = hi - lo;
        const int axis = size.x >= size.y && size.x >= size.z ? 0 : size.y >= size.z ? 1 : 2;
        // Split on a multiple of the piece size, so pieces come out full.
        const size_t half = (part.size() / 2 + leaf_triangles - 1) / leaf_triangles * leaf_triangles;
        std::nth_element(part.begin(), part.begin() + half, part.end(),
                         [&](uint32_t a, uint32_t b) { return centroid[a][axis] < centroid[b][axis]; });
        // Pushed in reverse so pieces come out in spatial order.
        stack.emplace_back(part.begin() + half, part.end());
        stack.emplace_back(part.begin(), part.begin() + half);
    }

    std::vector<std::vector<std::vector<uint32_t>>> results(pieces.size());
    parallel_for(pieces.size(), [&](size_t i) { results[i] = clusterize_piece(positions, indices, pieces[i]); });
    std::vector<std::vector<uint32_t>> out;
    for (auto& r : results)
        for (auto& c : r) out.push_back(std::move(c));
    return out;
}
