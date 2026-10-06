#include "geometry_file.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace {


}  // namespace

size_t geometry::leaf_triangles() const {
    size_t n = 0;
    for (const gpu_cluster& c : clusters)
        if (c.level == 0) n += c.triangle_count;
    return n;
}

geometry pack(const lod_mesh& lod) {
    geometry g;
    g.levels = lod.levels;
    g.positions.reserve(3 * lod.positions.size());
    for (const vec3& p : lod.positions) g.positions.insert(g.positions.end(), {p.x, p.y, p.z});
    for (const vec3& n : lod.normals) g.normals.insert(g.normals.end(), {n.x, n.y, n.z});
    g.skin_joints = lod.skin_joints;
    g.skin_weights = lod.skin_weights;
    g.vertex_source = lod.vertex_source;
    if (!lod.wedge_uvs.empty()) {
        g.wedge_vertex = lod.wedge_vertex;
        g.wedge_uvs = lod.wedge_uvs;
        g.wedge_material = lod.wedge_material;
        g.materials = lod.materials;
        vec2 lo = g.wedge_uvs[0], hi = lo;
        for (const vec2& uv : g.wedge_uvs) {
            lo = {std::min(lo.x, uv.x), std::min(lo.y, uv.y)};
            hi = {std::max(hi.x, uv.x), std::max(hi.y, uv.y)};
        }
        // a square range, at least 0 to 1 (an unrepeated texture keeps its 65535 steps).
        g.uv_min = {std::min(lo.x, 0.0f), std::min(lo.y, 0.0f)};
        g.uv_extent = std::max({hi.x - g.uv_min.x, hi.y - g.uv_min.y, 1.0f});
    }
    g.bounds = bounding_sphere(lod.positions.size(), [&](size_t i) { return lod.positions[i]; });
    // every lod sphere, which can stick out past the model.
    g.lod_bounds = g.bounds;
    for (const lod_cluster& c : lod.clusters) {
        g.lod_bounds = merge(g.lod_bounds, c.lod_bounds);
        if (!std::isinf(c.parent_error)) g.lod_bounds = merge(g.lod_bounds, c.parent_bounds);
    }

    // by parent error: a cluster is drawable only while its parent looks too
    // coarse, so from afar only a tail is, and the viewer skips the rest with
    // one binary search.
    std::vector<uint32_t> order(lod.clusters.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return lod.clusters[a].parent_error < lod.clusters[b].parent_error; });

    std::unordered_map<uint32_t, uint32_t> local;
    for (uint32_t ci : order) {
        const lod_cluster& c = lod.clusters[ci];
        gpu_cluster out{};
        auto put = [](float* dst, vec3 v) { dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; };
        put(out.center, c.bounds.center);
        out.radius = c.bounds.radius;
        put(out.cone_axis, c.cone_axis);
        out.cone_cutoff = c.cone_cutoff;
        put(out.lod_center, c.lod_bounds.center);
        out.lod_radius = c.lod_bounds.radius;
        put(out.parent_center, c.parent_bounds.center);
        out.parent_radius = c.parent_bounds.radius;
        out.lod_error = c.lod_error;
        out.parent_error = std::isinf(c.parent_error) ? FLT_MAX : c.parent_error;
        out.level = c.level;
        out.group = c.group;
        out.creator = c.creator;
        out.vertex_offset = static_cast<uint32_t>(g.cluster_vertices.size());
        out.triangle_offset = static_cast<uint32_t>(g.cluster_triangles.size());

        local.clear();
        for (size_t t = 0; t < c.indices.size(); t += 3) {
            uint32_t packed = 0;
            for (int k = 0; k < 3; ++k) {
                // untextured, the wedge is the vertex.
                const uint32_t w = g.textured() ? c.corners[t + k] : c.indices[t + k];
                auto [it, added] = local.emplace(w, static_cast<uint32_t>(local.size()));
                if (added) g.cluster_vertices.push_back(w);
                packed |= it->second << (8 * k);
            }
            g.cluster_triangles.push_back(packed);
        }
        out.vertex_count = static_cast<uint32_t>(local.size());
        out.triangle_count = static_cast<uint32_t>(c.indices.size() / 3);
        if (out.vertex_count > cluster_max_vertices || out.triangle_count > cluster_max_triangles)
            throw std::logic_error("cluster over the size limit");
        g.clusters.push_back(out);
    }
    return g;
}

geometry trim(const geometry& g, size_t max_triangles) {
    if (g.clusters.empty()) return g;  // nothing to trim, and no errors to search
    auto triangles_at = [&](float t) {
        size_t n = 0;
        for (const gpu_cluster& c : g.clusters)
            if (c.lod_error <= t && t < c.parent_error) n += c.triangle_count;
        return n;
    };
    std::vector<float> errors;
    for (const gpu_cluster& c : g.clusters) errors.push_back(c.lod_error);
    std::sort(errors.begin(), errors.end());
    errors.erase(std::unique(errors.begin(), errors.end()), errors.end());
    size_t lo = 0, hi = errors.size() - 1;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (triangles_at(errors[mid]) <= max_triangles) hi = mid;
        else lo = mid + 1;
    }
    const float floor_error = errors[lo];

    geometry out;
    out.bounds = g.bounds;
    out.lod_bounds = g.lod_bounds;
    out.materials = g.materials;
    out.uv_min = g.uv_min;
    out.uv_extent = g.uv_extent;
    std::vector<uint32_t> remap(g.positions.size() / 3, UINT32_MAX);
    std::vector<uint32_t> wedge_remap(g.textured() ? g.wedge_uvs.size() : 0, UINT32_MAX);
    for (const gpu_cluster& c : g.clusters) {
        if (c.parent_error <= floor_error) continue;  // finer than the new leaves everywhere
        gpu_cluster k = c;
        if (k.lod_error <= floor_error) {
            k.lod_error = 0;
            k.level = 0;
            k.creator = UINT32_MAX;  // its source is gone
        }
        k.vertex_offset = static_cast<uint32_t>(out.cluster_vertices.size());
        k.triangle_offset = static_cast<uint32_t>(out.cluster_triangles.size());
        for (uint32_t i = 0; i < c.vertex_count; ++i) {
            const uint32_t w = g.cluster_vertices[c.vertex_offset + i];
            uint32_t& v = remap[g.vertex_of(w)];
            if (v == UINT32_MAX) {
                v = static_cast<uint32_t>(out.positions.size() / 3);
                const size_t src = 3 * size_t(g.vertex_of(w));
                out.positions.insert(out.positions.end(), &g.positions[src], &g.positions[src + 3]);
                out.normals.insert(out.normals.end(), &g.normals[src], &g.normals[src + 3]);
                if (g.skinned()) {
                    const size_t sv = 4 * size_t(g.vertex_of(w));
                    out.skin_joints.insert(out.skin_joints.end(), &g.skin_joints[sv], &g.skin_joints[sv + 4]);
                    out.skin_weights.insert(out.skin_weights.end(), &g.skin_weights[sv], &g.skin_weights[sv + 4]);
                }
            }
            if (!g.textured()) {
                out.cluster_vertices.push_back(v);
                continue;
            }
            uint32_t& nw = wedge_remap[w];
            if (nw == UINT32_MAX) {
                nw = static_cast<uint32_t>(out.wedge_uvs.size());
                out.wedge_uvs.push_back(g.wedge_uvs[w]);
                out.wedge_vertex.push_back(v);
                out.wedge_material.push_back(g.wedge_material[w]);
            }
            out.cluster_vertices.push_back(nw);
        }
        out.cluster_triangles.insert(out.cluster_triangles.end(), g.cluster_triangles.begin() + c.triangle_offset,
                                     g.cluster_triangles.begin() + c.triangle_offset + c.triangle_count);
        out.clusters.push_back(k);
    }
    // levels renumbered from the new leaves; stats kept for those left.
    uint32_t lowest = UINT32_MAX;
    for (const gpu_cluster& c : out.clusters)
        if (c.level != 0) lowest = std::min(lowest, c.level);
    if (lowest != UINT32_MAX)
        for (gpu_cluster& c : out.clusters)
            if (c.level != 0) c.level -= lowest - 1;
    out.levels = g.levels;
    if (lowest != UINT32_MAX && lowest > 1) out.levels.erase(out.levels.begin(), out.levels.begin() + (lowest - 1));
    // still by parent error: order kept, some removed.
    return out;
}
