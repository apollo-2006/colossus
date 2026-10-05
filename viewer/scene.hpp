#pragma once
// the scene: models' clusters, pages, textures and skeletons read from disk, shadow copies cut
// from them, and the instances placed on a grid in cells.
#include "gpu_types.hpp"
#include "options.hpp"
#include "paged_file.hpp"
#include "skeleton.hpp"
#include "streamer.hpp"
#include "texture_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace viewer {

// pose slots: a skinned instance follows one of a model's animations at one of this many
// phases, so a crowd shares a few poses, each posed once a frame.
constexpr uint32_t pose_phases = 16;

// the largest stretch of a 3x3 matrix (rows of a 3x4), its spectral norm, less the identity
// first if asked: power iteration on m'm.
float spectral_norm(const float* rows, bool less_identity) {
    float a[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) a[r][c] = rows[4 * r + c] - (less_identity && r == c ? 1.0f : 0.0f);
    float ata[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) ata[i][j] += a[k][i] * a[k][j];
    float v[3] = {0.577f, 0.577f, 0.577f}, lambda = 0;
    for (int it = 0; it < 32; ++it) {
        float w[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) w[i] += ata[i][j] * v[j];
        const float l = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (l < 1e-20f) return 0;
        lambda = l;
        for (int i = 0; i < 3; ++i) v[i] = w[i] / l;
    }
    // power iteration approaches from below: a little over, to stay a bound.
    return std::sqrt(lambda) * 1.01f + 1e-6f;
}

// coarser copy of a model for shadow rays.
struct shadow_mesh {
    uint32_t first_vertex, vertex_count;  // in scene::shadow_positions
    uint32_t first_index, index_count;    // in scene::shadow_indices, relative to first_vertex
};

struct scene {
    std::vector<packed_cluster> clusters;  // every model's, page numbers made global
    std::vector<page_bounds> shared_bounds;  // per global page
    std::vector<stream_page> pages;
    std::vector<uint32_t> deps;         // global page numbers
    std::vector<uint32_t> children;     // global page numbers, for prefetch
    std::vector<int> files;
    std::vector<gpu_mesh> meshes;
    std::vector<gpu_instance> instances;
    std::vector<gpu_cell> cells;
    std::vector<shadow_mesh> shadow_meshes;  // shadow_lods per model
    float shadow_lod_error[4] = {};          // each shadow copy's largest error over the models
    std::vector<float> shadow_positions;
    std::vector<uint32_t> shadow_indices;
    // skinning: each skinned model's skeleton, the pose slots, and per global page its joints
    // and weight spread (zero for unskinned pages).
    struct pose_slot {
        uint32_t skeleton;
        int32_t animation;  // -1: the rest pose
        float phase;        // of the animation's duration
        uint32_t mesh;
    };
    std::vector<skeleton> skeletons;
    std::vector<pose_slot> pose_slots;
    uint32_t pose_joints = 0;              // all slots' joints
    std::vector<page_skin> page_skins;
    std::vector<float> skin_reach;         // per model: the farthest any pose moves its sphere
    size_t instanced_triangles = 0;  // at full detail, all instances
    size_t moving = 0;               // moving instances
    uint64_t total_page_bytes = 0;

    scene() = default;
    scene(const scene&) = delete;
    ~scene() {
        for (int fd : files) close(fd);
    }
};

// finest cut within the shadow budget, from the model's pages: positions (three floats a vertex,
// per cluster) and indices. returns its error.
float shadow_cut(const paged_geometry& g, int fd, size_t budget, std::vector<float>& positions, std::vector<uint32_t>& indices) {
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
    size_t lo = 0, hi = errors.size() - 1;  // the coarsest cut is the roots: small enough
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (triangles_at(errors[mid]) <= budget) hi = mid;
        else lo = mid + 1;
    }
    const float error = errors[lo];
    std::vector<uint32_t> words;
    uint32_t loaded_page = no_page;
    for (const gpu_cluster& c : g.clusters) {
        if (!(c.lod_error <= error && error < c.parent_error)) continue;
        if (c.group != loaded_page) {
            const page_info& pg = g.pages[c.group];
            words.resize(pg.size / 4);
            if (pread(fd, words.data(), pg.size, static_cast<off_t>(g.data_offset + pg.offset)) != static_cast<ssize_t>(pg.size))
                throw std::runtime_error("reading a page failed");
            loaded_page = c.group;
        }
        const uint32_t first = static_cast<uint32_t>(positions.size() / 3);
        for (uint32_t k = 0; k < c.vertex_count; ++k) {
            const vec3 p = decode_position(g, c, words.data(), k);
            positions.insert(positions.end(), {p.x, p.y, p.z});
        }
        for (uint32_t t = 0; t < c.triangle_count; ++t) {
            const uint32_t w = decode_triangle(c, words.data(), t);
            for (int k = 0; k < 3; ++k) indices.push_back(first + ((w >> (8 * k)) & 255));
        }
    }
    return error;
}

// reads models' clusters and page lists (geometry streams later) and places them on a grid,
// turned and sized at random.
void make_scene(const options& opt, scene& s) {
    std::vector<size_t> leaf_triangles;
    for (const std::string& path : opt.models) {
        const paged_geometry g = load_paged(path, false);
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        if (opt.cold) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        s.files.push_back(fd);
        std::printf("%s: %zu clusters, %zu triangles at full detail, %zu pages, %.0f MB\n", path.c_str(), g.clusters.size(),
                    g.leaf_triangles(), g.pages.size(), g.data_size / 1048576.0);
        leaf_triangles.push_back(g.leaf_triangles());
        s.total_page_bytes += g.data_size;

        const uint32_t page_base = static_cast<uint32_t>(s.pages.size());
        const uint32_t dep_base = static_cast<uint32_t>(s.deps.size());
        for (uint32_t d : g.deps) s.deps.push_back(d + page_base);
        for (uint32_t p = 0; p < g.pages.size(); ++p) {
            const page_info& pg = g.pages[p];
            s.pages.push_back({fd, g.data_offset + pg.offset, pg.size, pg.dep_first + dep_base, pg.dep_count, p == 0});
        }
        // each page's children: the finer pages its clusters stand for. its error, for prefetch.
        std::vector<std::vector<uint32_t>> children(g.pages.size());
        for (const gpu_cluster& c : g.clusters) {
            if (c.creator != no_page) children[c.group].push_back(c.creator);
            float& e = s.pages[page_base + c.group].error;
            e = std::max(e, c.parent_error);
        }
        for (uint32_t p = 0; p < g.pages.size(); ++p) {
            auto& ch = children[p];
            std::sort(ch.begin(), ch.end());
            ch.erase(std::unique(ch.begin(), ch.end()), ch.end());
            stream_page& sp = s.pages[page_base + p];
            sp.child_first = static_cast<uint32_t>(s.children.size());
            sp.child_count = static_cast<uint32_t>(ch.size());
            for (uint32_t c : ch) s.children.push_back(c + page_base);
        }

        float shadow_error = 0;
        for (uint32_t lod = 0; lod < shadow_lods; ++lod) {
            shadow_mesh sm{};
            sm.first_vertex = static_cast<uint32_t>(s.shadow_positions.size() / 3);
            sm.first_index = static_cast<uint32_t>(s.shadow_indices.size());
            std::vector<float> positions;
            std::vector<uint32_t> indices;
            const float error = shadow_cut(g, fd, shadow_budgets[lod], positions, indices);
            if (lod == 0) shadow_error = error;
            sm.vertex_count = static_cast<uint32_t>(positions.size() / 3);
            sm.index_count = static_cast<uint32_t>(indices.size());
            s.shadow_positions.insert(s.shadow_positions.end(), positions.begin(), positions.end());
            s.shadow_indices.insert(s.shadow_indices.end(), indices.begin(), indices.end());
            s.shadow_meshes.push_back(sm);
            s.shadow_lod_error[lod] = std::max(s.shadow_lod_error[lod], error);
            std::printf("  shadow copy %u: %u triangles, error %.3g\n", lod, sm.index_count / 3, error);
        }

        gpu_mesh m{};
        // a texture beside the model streams with it: its tiles are pages too, each depending
        // on the coarser tile over it, so a resident tile's coarser copies are resident.
        const std::string ctex = path.substr(0, path.find_last_of('.')) + ".ctex";
        if (access(ctex.c_str(), R_OK) == 0) {
            const texture_info t = load_texture_info(ctex);
            const int tfd = open(ctex.c_str(), O_RDONLY);
            if (tfd < 0) throw std::runtime_error("cannot open " + ctex);
            if (opt.cold) posix_fadvise(tfd, 0, 0, POSIX_FADV_DONTNEED);
            s.files.push_back(tfd);
            const uint32_t first = static_cast<uint32_t>(s.pages.size());
            for (uint32_t k = 0; k < t.tile_count(); ++k) {
                const uint32_t parent = t.parent(k);
                const uint32_t dep_first = static_cast<uint32_t>(s.deps.size());
                if (parent != UINT32_MAX) s.deps.push_back(first + parent);
                stream_page sp{tfd, t.data_offset + uint64_t(k) * tile_bytes, tile_bytes, dep_first,
                               parent != UINT32_MAX ? 1u : 0u, parent == UINT32_MAX};
                sp.child_first = static_cast<uint32_t>(s.children.size());
                s.pages.push_back(sp);
            }
            m.texture[0] = first;
            m.texture[1] = static_cast<uint32_t>(t.levels.size());
            m.texture[2] = t.width;
            m.texture[3] = t.height;
            s.total_page_bytes += uint64_t(t.tile_count()) * tile_bytes;
            std::printf("  texture %u x %u: %zu levels, %u tiles, %.0f MB\n", t.width, t.height, t.levels.size(), t.tile_count(),
                        t.tile_count() * double(tile_bytes) / 1048576.0);
        }
        // a skeleton beside the model makes it skinned: its pages' skin bounds, and pose slots
        // (every animation at pose_phases phases).
        const std::string cskn = path.substr(0, path.find_last_of('.')) + ".cskn";
        float reach = 0;
        if (access(cskn.c_str(), R_OK) == 0) {
            skeleton sk = load_skeleton(cskn);
            if (sk.pages.size() != g.pages.size()) throw std::runtime_error(cskn + " does not match " + path + ": rebuild both");
            if (s.page_skins.size() < page_base + g.pages.size()) s.page_skins.resize(page_base + g.pages.size());
            std::copy(sk.pages.begin(), sk.pages.end(), s.page_skins.begin() + page_base);
            const uint32_t joints = static_cast<uint32_t>(sk.joints.size());
            m.skin[0] = joints;
            m.skin[1] = static_cast<uint32_t>(s.pose_slots.size());
            m.skin[2] = s.pose_joints;
            const int anims = static_cast<int>(sk.animations.size());
            for (int a = anims ? 0 : -1; a < std::max(anims, 0); ++a)
                for (uint32_t ph = 0; ph < (a < 0 ? 1u : pose_phases); ++ph) {
                    s.pose_slots.push_back({uint32_t(s.skeletons.size()), a, float(ph) / pose_phases, uint32_t(s.meshes.size())});
                    s.pose_joints += joints;
                }
            m.skin[3] = sk.anchor;
            // the farthest any pose moves the model's sphere, sampled finely and given a tenth
            // more: cells (instance groups) are culled by it.
            std::vector<float> rows;
            const vec3 c = g.bounds.center;
            for (int a = anims ? 0 : -1; a < std::max(anims, 0); ++a)
                for (int k = 0; k < 256; ++k) {
                    pose_joints(sk, a, a < 0 ? 0 : sk.animations[size_t(a)].duration * k / 256, rows);
                    for (uint32_t j = 0; j < joints; ++j) {
                        const float* r = &rows[12 * j];
                        const vec3 moved(r[0] * c.x + r[1] * c.y + r[2] * c.z + r[3], r[4] * c.x + r[5] * c.y + r[6] * c.z + r[7],
                                         r[8] * c.x + r[9] * c.y + r[10] * c.z + r[11]);
                        reach = std::max(reach, length(moved - c) + spectral_norm(r, true) * g.bounds.radius);
                    }
                }
            reach *= 1.1f;
            std::printf("  skeleton: %u joints, %d animations", joints, anims);
            for (const animation& an : sk.animations) std::printf(" %s (%.2fs)", an.name.c_str(), an.duration);
            std::printf(", reach %.3g of radius %.3g\n", reach, g.bounds.radius);
            s.skeletons.push_back(std::move(sk));
        }
        s.skin_reach.push_back(reach);
        m.shadow_error = shadow_error;
        m.first_cluster = static_cast<uint32_t>(s.clusters.size());
        m.cluster_count = static_cast<uint32_t>(g.clusters.size());
        m.bounds[0] = g.bounds.center.x; m.bounds[1] = g.bounds.center.y; m.bounds[2] = g.bounds.center.z;
        m.bounds[3] = g.bounds.radius;
        m.lod_bounds[0] = g.lod_bounds.center.x; m.lod_bounds[1] = g.lod_bounds.center.y;
        m.lod_bounds[2] = g.lod_bounds.center.z; m.lod_bounds[3] = g.lod_bounds.radius;
        m.grid[0] = g.grid_min.x; m.grid[1] = g.grid_min.y; m.grid[2] = g.grid_min.z; m.grid[3] = g.grid_step;
        s.meshes.push_back(m);
        for (packed_cluster c : g.packed) {
            c.group += page_base;
            if (c.creator != no_page) c.creator += page_base;
            s.clusters.push_back(c);
        }
        // per global page: the previous model's texture tiles have none.
        s.shared_bounds.resize(page_base);
        s.shared_bounds.insert(s.shared_bounds.end(), g.shared_bounds.begin(), g.shared_bounds.end());
    }
    s.shared_bounds.resize(s.pages.size());

    s.page_skins.resize(std::max<size_t>(s.pages.size(), 1));  // texture pages and unskinned models: none
    std::mt19937 rng(7), material_rng(11), motion_rng(13);
    std::uniform_real_distribution<float> chance(0, 1);
    std::uniform_real_distribution<float> turn(0, 6.2831853f), size(0.85f, 1.15f);
    // mostly marble; some sandstone, bronze, gold, granite (shade.comp's table). a lone statue
    // is marble.
    const uint32_t mix[20] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5};
    const int n = std::max(1, opt.grid);
    // tiles of cell_side x cell_side so cells are consecutive; random draws stay in row order.
    std::vector<gpu_instance> placed(size_t(n) * n);
    for (int z = 0; z < n; ++z)
        for (int x = 0; x < n; ++x) {
            const uint32_t mesh = static_cast<uint32_t>((z * n + x) % s.meshes.size());
            const float scale = n == 1 ? 1.0f : size(rng);
            const float angle = n == 1 ? 0.0f : turn(rng);
            const vec3 at = {(x - (n - 1) * 0.5f) * opt.spacing, 0, (z - (n - 1) * 0.5f) * -opt.spacing};
            const mat4 m = trs(at, {0, 1, 0}, angle, scale);
            gpu_instance inst{};
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c) inst.rows[r][c] = m.at(r, c);
            inst.mesh = mesh;
            inst.scale = scale;
            inst.material = !opt.mixed_materials ? 0 : n == 1 ? 1 : mix[material_rng() % 20];
            if (opt.material >= 0) inst.material = uint32_t(opt.material);
            // moving: bit 9 marks it, bit 8 the direction, low 8 bits the phase. deforming: bit 10
            // (deform() in common.glsl), shadows and motion vectors as for movers.
            if (chance(motion_rng) < opt.moving) inst.anim = 512u | (motion_rng() & 511u);
            if (opt.deforming > 0 && chance(motion_rng) < opt.deforming) inst.anim |= 1024u | (inst.anim ? 0u : motion_rng() & 255u);
            // skinned: an animation (the named one, or one at random) at a random phase; bit 11
            // and the pose slot from bit 16 (common.glsl). no sway on top.
            if (const uint32_t* sk = s.meshes[mesh].skin; sk[0]) {
                const skeleton& skel = s.skeletons[s.pose_slots[sk[1]].skeleton];
                const uint32_t anims = std::max<uint32_t>(uint32_t(skel.animations.size()), 1);
                uint32_t a = motion_rng() % anims;
                for (uint32_t k = 0; k < skel.animations.size(); ++k)
                    if (skel.animations[k].name == opt.animation) a = k;
                const uint32_t phases = skel.animations.empty() ? 1 : pose_phases;
                const uint32_t slot = sk[1] + a * phases + motion_rng() % phases;
                inst.anim = (inst.anim & ~1024u) | 2048u | slot << 16;
            }
            s.moving += inst.anim != 0;
            placed[size_t(z) * n + x] = inst;
            s.instanced_triangles += leaf_triangles[mesh];
        }
    for (int tz = 0; tz < n; tz += cell_side)
        for (int tx = 0; tx < n; tx += cell_side) {
            gpu_cell cell{};
            cell.first = uint32_t(s.instances.size());
            // each instance's sphere: its bounds, or for a moving one its whole path (turns
            // about its origin, drifts 0.2: animate() in common.glsl).
            std::vector<std::pair<vec3, float>> spheres;
            for (int z = tz; z < std::min(n, tz + int(cell_side)); ++z)
                for (int x = tx; x < std::min(n, tx + int(cell_side)); ++x) {
                    const gpu_instance& inst = placed[size_t(z) * n + x];
                    const gpu_mesh& m = s.meshes[inst.mesh];
                    const vec3 local = {m.bounds[0], m.bounds[1], m.bounds[2]};
                    const vec3 origin = {inst.rows[0][3], inst.rows[1][3], inst.rows[2][3]};
                    vec3 c = origin;
                    for (int r = 0; r < 3; ++r)
                        (&c.x)[r] += inst.rows[r][0] * local.x + inst.rows[r][1] * local.y + inst.rows[r][2] * local.z;
                    // deforming: points move up to the reach (deform_reach in common.glsl).
                    float radius = (m.bounds[3] * ((inst.anim & 1024u) ? 1.1f : 1.0f) + s.skin_reach[inst.mesh]) * inst.scale;
                    if (inst.anim & 512u) {
                        radius += length(c - origin) + 0.2f;
                        c = origin;
                    }
                    spheres.push_back({c, radius});
                    s.instances.push_back(inst);
                }
            cell.count = uint32_t(s.instances.size()) - cell.first;
            vec3 lo = spheres[0].first, hi = lo;
            for (const auto& [c, r] : spheres) {
                lo = min(lo, c);
                hi = max(hi, c);
            }
            const vec3 center = (lo + hi) * 0.5f;
            float radius = 0;
            for (const auto& [c, r] : spheres) radius = std::max(radius, length(c - center) + r);
            cell.center[0] = center.x; cell.center[1] = center.y; cell.center[2] = center.z;
            cell.radius = radius;
            s.cells.push_back(cell);
        }
}

}  // namespace viewer
