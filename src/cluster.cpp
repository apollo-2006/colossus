#include "cluster.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr size_t leaf_triangles = 8192;

// Recursive bisection of the piece's triangle graph (triangles joined
// across edges): each split grows one half by breadth-first search from a
// far-off triangle until it holds a whole number of clusters' worth, and
// the rest is the other half. Halves grown that way are compact (balls in
// the graph), and every cluster but one per piece comes out full.
std::vector<std::vector<uint32_t>> bisect_piece(const std::vector<vec3>& positions, const std::vector<uint32_t>& indices,
                                                const std::vector<uint32_t>& tris) {
    const size_t n = tris.size();
    std::vector<std::vector<uint32_t>> out;
    if (n == 0) return out;
    std::vector<uint32_t> verts;
    verts.reserve(3 * n);
    for (uint32_t t : tris)
        for (int c = 0; c < 3; ++c) verts.push_back(indices[3 * t + c]);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    const size_t nv = verts.size();
    std::vector<uint32_t> corner(3 * n);
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
    // Neighbours across edges, and across vertices as a fallback for
    // pieces of surface that only touch at a point.
    std::vector<uint32_t> neighbour(3 * n, UINT32_MAX);
    for (size_t i = 0; i < n; ++i)
        for (int e = 0; e < 3; ++e) {
            const uint32_t a = corner[3 * i + e], b = corner[3 * i + (e + 1) % 3];
            for (uint32_t k = adj_start[a]; k < adj_start[a + 1]; ++k) {
                const uint32_t u = adj[k];
                if (u == i) continue;
                const uint32_t* cu = &corner[3 * u];
                if (cu[0] == b || cu[1] == b || cu[2] == b) {
                    neighbour[3 * i + e] = u;
                    break;
                }
            }
        }
    std::vector<vec3> centroid(n);
    for (size_t i = 0; i < n; ++i)
        centroid[i] = (positions[verts[corner[3 * i]]] + positions[verts[corner[3 * i + 1]]] + positions[verts[corner[3 * i + 2]]]) *
                      (1.0f / 3);

    std::vector<uint32_t> member(n, UINT32_MAX), seen(n, UINT32_MAX), vertex_stamp(nv, UINT32_MAX);
    uint32_t stamp = 0;
    // Breadth-first order of `set` from `seed`; pieces the search cannot
    // reach are taken up nearest the seed first.
    auto bfs = [&](const std::vector<uint32_t>& set, uint32_t set_id, uint32_t seed, std::vector<uint32_t>& order) {
        ++stamp;
        order.clear();
        size_t head = 0;
        std::vector<uint32_t> rest;  // Unreached members, by distance, for when the search runs dry
        bool rest_sorted = false;
        size_t rest_next = 0;
        auto push = [&](uint32_t t) {
            if (member[t] == set_id && seen[t] != stamp) {
                seen[t] = stamp;
                order.push_back(t);
            }
        };
        push(seed);
        while (order.size() < set.size()) {
            if (head == order.size()) {
                if (!rest_sorted) {
                    rest = set;
                    const vec3 c = centroid[seed];
                    std::sort(rest.begin(), rest.end(),
                              [&](uint32_t a, uint32_t b) { return length(centroid[a] - c) < length(centroid[b] - c); });
                    rest_sorted = true;
                }
                while (seen[rest[rest_next]] == stamp) ++rest_next;
                push(rest[rest_next]);
            }
            const uint32_t t = order[head++];
            for (int e = 0; e < 3; ++e)
                if (neighbour[3 * t + e] != UINT32_MAX) push(neighbour[3 * t + e]);
            for (int c = 0; c < 3; ++c) {
                const uint32_t v = corner[3 * t + c];
                for (uint32_t k = adj_start[v]; k < adj_start[v + 1]; ++k) push(adj[k]);
            }
        }
    };
    auto vertex_count = [&](const std::vector<uint32_t>& set) {
        ++stamp;
        uint32_t count = 0;
        for (uint32_t t : set)
            for (int c = 0; c < 3; ++c)
                if (vertex_stamp[corner[3 * t + c]] != stamp) {
                    vertex_stamp[corner[3 * t + c]] = stamp;
                    ++count;
                }
        return count;
    };

    uint32_t next_id = 0;
    std::vector<uint32_t> order, far_order, level_a, level_b;
    std::vector<std::vector<uint32_t>> stack(1);
    for (uint32_t i = 0; i < n; ++i) stack[0].push_back(i);
    while (!stack.empty()) {
        std::vector<uint32_t> set = std::move(stack.back());
        stack.pop_back();
        if (set.size() <= cluster_max_triangles && vertex_count(set) <= cluster_max_vertices) {
            for (uint32_t& t : set) t = tris[t];
            out.push_back(std::move(set));
            continue;
        }
        const uint32_t id = next_id++;
        for (uint32_t t : set) member[t] = id;
        // Two far-apart seeds: the last triangle a search from any member
        // reaches, and the last one a search from there reaches. Each
        // triangle leans to the seed it is fewer steps from; the halves are
        // cut at a whole number of clusters along that lean, which keeps
        // both of them compact. (Growing one half from one seed and taking
        // the rest as the other half left the rest ragged, and its
        // clusters simplified badly: coarse levels came out with ten times
        // the error.)
        bfs(set, id, set[0], far_order);
        const uint32_t a = far_order.back();
        bfs(set, id, a, order);
        const uint32_t b = order.back();
        // Steps from a seed: breadth-first, across edges, then across
        // shared vertices, a level each; a triangle the search cannot
        // reach is a level past the last one reached.
        auto levels = [&](uint32_t seed, std::vector<uint32_t>& level) {
            ++stamp;
            std::vector<uint32_t> queue;
            queue.reserve(set.size());
            auto reach = [&](uint32_t t, uint32_t l) {
                if (member[t] == id && seen[t] != stamp) {
                    seen[t] = stamp;
                    level[t] = l;
                    queue.push_back(t);
                }
            };
            reach(seed, 0);
            size_t head = 0, unreached = 0;
            std::vector<uint32_t> rest;
            while (queue.size() < set.size()) {
                if (head == queue.size()) {
                    if (rest.empty()) {
                        rest = set;
                        const vec3 c = centroid[seed];
                        std::sort(rest.begin(), rest.end(),
                                  [&](uint32_t x, uint32_t y) { return length(centroid[x] - c) < length(centroid[y] - c); });
                    }
                    while (seen[rest[unreached]] == stamp) ++unreached;
                    reach(rest[unreached], level[queue.back()] + 1);
                }
                const uint32_t t = queue[head++];
                for (int e = 0; e < 3; ++e)
                    if (neighbour[3 * t + e] != UINT32_MAX) reach(neighbour[3 * t + e], level[t] + 1);
                for (int c = 0; c < 3; ++c) {
                    const uint32_t v = corner[3 * t + c];
                    for (uint32_t k = adj_start[v]; k < adj_start[v + 1]; ++k) reach(adj[k], level[t] + 2);
                }
            }
        };
        if (level_a.size() < n) {
            level_a.resize(n);
            level_b.resize(n);
        }
        levels(a, level_a);
        levels(b, level_b);
        const vec3 ca = centroid[a], cb = centroid[b];
        auto lean = [&](uint32_t t) {
            return float(int(level_a[t]) - int(level_b[t])) + 1e-3f * (length(centroid[t] - ca) - length(centroid[t] - cb));
        };
        order = set;
        const size_t whole = (set.size() + cluster_max_triangles - 1) / cluster_max_triangles;
        // Split so both halves hold whole clusters; a set that is one
        // cluster too many vertices is halved.
        const size_t left = whole >= 2 ? whole / 2 * cluster_max_triangles : set.size() / 2;
        std::nth_element(order.begin(), order.begin() + left, order.end(), [&](uint32_t x, uint32_t y) { return lean(x) < lean(y); });
        // Smooth the cut: a triangle with more edge neighbours on the other
        // side changes sides, paired with one that wants to go the other
        // way, so the sizes hold. A few passes shorten the outline, and a
        // shorter outline locks fewer edges when groups are simplified.
        {
            const uint32_t left_id = next_id++;
            for (size_t i = 0; i < left; ++i) member[order[i]] = left_id;
            auto pull = [&](uint32_t t, uint32_t side) {  // Edge neighbours on `side` minus on the other
                int d = 0;
                for (int e = 0; e < 3; ++e) {
                    const uint32_t u = neighbour[3 * t + e];
                    if (u == UINT32_MAX) continue;
                    if (member[u] == side) ++d;
                    else if (member[u] == (side == left_id ? id : left_id)) --d;
                }
                return d;
            };
            for (int pass = 0; pass < 4; ++pass) {
                std::vector<uint32_t> to_right, to_left;
                for (size_t i = 0; i < order.size(); ++i) {
                    const uint32_t t = order[i];
                    if (member[t] == left_id && pull(t, id) > 0) to_right.push_back(t);
                    else if (member[t] == id && pull(t, left_id) > 0) to_left.push_back(t);
                }
                const size_t swaps = std::min(to_right.size(), to_left.size());
                if (swaps == 0) break;
                for (size_t k = 0; k < swaps; ++k) {
                    member[to_right[k]] = id;
                    member[to_left[k]] = left_id;
                }
            }
            size_t l = 0;
            std::vector<uint32_t> right_part;
            for (uint32_t t : order) {
                if (member[t] == left_id) order[l++] = t;
                else right_part.push_back(t);
            }
            std::copy(right_part.begin(), right_part.end(), order.begin() + l);
        }
        stack.emplace_back(order.begin() + left, order.end());
        stack.emplace_back(order.begin(), order.begin() + left);
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
    parallel_for(pieces.size(), [&](size_t i) { results[i] = bisect_piece(positions, indices, pieces[i]); });
    std::vector<std::vector<uint32_t>> out;
    for (auto& r : results)
        for (auto& c : r) out.push_back(std::move(c));
    return out;
}
