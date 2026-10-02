#include "lod_check.hpp"

#include <algorithm>
#include <cmath>

namespace {

struct tri_edge {
    uint32_t a, b;
    bool operator<(const tri_edge& o) const { return a != o.a ? a < o.a : b < o.b; }
    bool operator==(const tri_edge& o) const { return a == o.a && b == o.b; }
};

// edges used by exactly one of the triangles, and the area.
template <class ForEachTriangle>
std::vector<tri_edge> single_edges(ForEachTriangle&& each, const geometry& g, double& area) {
    std::vector<tri_edge> e;
    area = 0;
    each([&](uint32_t a, uint32_t b, uint32_t c) {
        const uint32_t v[3] = {a, b, c};
        for (int k = 0; k < 3; ++k) e.push_back({std::min(v[k], v[(k + 1) % 3]), std::max(v[k], v[(k + 1) % 3])});
        auto p = [&](uint32_t i) { return vec3(g.positions[3 * i], g.positions[3 * i + 1], g.positions[3 * i + 2]); };
        area += 0.5 * length(cross(p(b) - p(a), p(c) - p(a)));
    });
    std::sort(e.begin(), e.end());
    std::vector<tri_edge> out;
    for (size_t k = 0; k < e.size();) {
        size_t j = k;
        while (j < e.size() && e[j] == e[k]) ++j;
        if (j - k == 1) out.push_back(e[k]);
        k = j;
    }
    return out;
}

template <class Pick>
auto cluster_triangles(const geometry& g, Pick&& pick) {
    return [&g, pick](auto&& emit) {
        for (const gpu_cluster& c : g.clusters) {
            if (!pick(c)) continue;
            for (uint32_t t = 0; t < c.triangle_count; ++t) {
                const uint32_t p = g.cluster_triangles[c.triangle_offset + t];
                const uint32_t* v = &g.cluster_vertices[c.vertex_offset];
                emit(v[p & 255], v[p >> 8 & 255], v[p >> 16 & 255]);
            }
        }
    };
}

}  // namespace

std::vector<cut_report> check_cuts(const geometry& g, int steps) {
    double original_area = 0;
    const auto original_border =
        single_edges(cluster_triangles(g, [](const gpu_cluster& c) { return c.level == 0; }), g, original_area);
    std::vector<uint8_t> on_border(g.positions.size() / 3, 0);
    for (const tri_edge& e : original_border) on_border[e.a] = on_border[e.b] = 1;

    float top = 0;
    for (const gpu_cluster& c : g.clusters) top = std::max(top, c.lod_error);
    std::vector<cut_report> out;
    for (int s = 0; s <= steps; ++s) {
        // geometric spacing, plus 0 and one past the top.
        const float t = s == 0 ? 0 : top * 1.01f * std::pow(1e-4f, 1 - static_cast<float>(s) / steps);
        cut_report r;
        r.threshold = t;
        auto pick = [t](const gpu_cluster& c) { return c.lod_error <= t && t < c.parent_error; };
        for (const gpu_cluster& c : g.clusters)
            if (pick(c)) {
                ++r.clusters;
                r.triangles += c.triangle_count;
            }
        double area = 0;
        for (const tri_edge& e : single_edges(cluster_triangles(g, pick), g, area))
            if (!on_border[e.a] || !on_border[e.b]) ++r.cracked_edges;
        r.area_ratio = original_area > 0 ? area / original_area : 0;
        out.push_back(r);
    }
    return out;
}
