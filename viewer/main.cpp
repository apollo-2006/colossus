// nexus_geometry viewer: draws instanced cluster LOD hierarchies (.ngeo
// files from ngeo_build) with task and mesh shaders on Vulkan.
//
// Each frame:
//   1. instance_cull.comp drops instances outside the view and, for each
//      one left, finds by binary search the clusters that might be drawn
//      at its distance, as work items of 64 clusters.
//   2. cluster_cull.comp tests each of those clusters: the LOD cut, the
//      frustum, the normal cone and occlusion. Survivors are appended to a
//      visible list.
//   3. draw.mesh draws each survivor; vis.frag writes depth and triangle
//      into a 64-bit visibility buffer with an atomic max. Clusters small
//      on screen go to sw_raster.comp instead, a compute rasterizer that
//      writes the same buffer the same way.
// Steps 1 to 3 run twice: first testing occlusion against last frame's
// depth pyramid, then, after a pyramid is built from what that drew,
// re-testing what the first pass found hidden.
//   4. shade.comp rebuilds the triangle under every pixel and shades it.
//
//   nexus_view --model models/lucy.ngeo --grid 10
//   nexus_view --model a.ngeo --model b.ngeo --grid 40 --headless --frames 60 --screenshot out.png
#include "geometry_file.hpp"
#include "png.hpp"
#include "vk.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

const uint32_t instance_cull_spv[] = {
#include "instance_cull.comp.inc"
};
const uint32_t args_spv[] = {
#include "args.comp.inc"
};
const uint32_t cluster_cull_spv[] = {
#include "cluster_cull.comp.inc"
};
const uint32_t draw_mesh_spv[] = {
#include "draw.mesh.inc"
};
const uint32_t vis_frag_spv[] = {
#include "vis.frag.inc"
};
const uint32_t shade_spv[] = {
#include "shade.comp.inc"
};
const uint32_t shade_rt_spv[] = {
#include "shade_rt.comp.inc"
};
const uint32_t sw_raster_spv[] = {
#include "sw_raster.comp.inc"
};
const uint32_t hzb_spv[] = {
#include "hzb.comp.inc"
};

constexpr uint32_t frames_in_flight = 2;
constexpr uint32_t max_work_items = 1u << 22;
constexpr uint32_t max_visible = 1u << 22;  // Leaves 7 bits for the triangle in a 32-bit id... and 3 to spare
constexpr const char* mode_names[] = {"shaded", "clusters", "triangles", "LOD level", "groups", "instances", "holes", "rasterizer"};
constexpr uint32_t mode_count = 8;

// Laid out as viewer/shaders/common.glsl declares them (scalar layout).
struct gpu_mesh {
    uint32_t first_cluster, cluster_count;
    float shadow_error;
    uint32_t pad1;
    float bounds[4];
    float lod_bounds[4];
};
struct gpu_instance {
    float rows[3][4];
    uint32_t mesh;
    float scale;
    uint32_t pad0, pad1;
};
struct gpu_frame {
    float view_proj[16];
    float cull_planes[5][4];
    float cull_origin[4];
    float origin[4];
    float inv_view_proj[16];
    uint32_t width, height;
    uint32_t instance_count;
    uint32_t flags;
    float lod_scale;
    float lod_threshold;
    float near_z;
    uint32_t debug_mode;
    uint32_t max_work_items;
    uint32_t max_visible;
    float time;
    uint32_t pad;
    float view[16];
    float prev_view[16];
    float p00, p11;
    uint32_t hzb_width, hzb_height, hzb_levels;
    float sw_max_pixels;
};
struct gpu_stats {
    uint32_t instances_visible, work_items, clusters_tested, clusters_drawn, triangles_drawn;
    uint32_t work_overflow, visible_overflow;
    uint32_t instances_occluded, clusters_occluded, clusters_late, clusters_software;
};
struct gpu_push {
    uint32_t pass, level;
};

constexpr uint32_t flag_cone_culling = 1, flag_frustum_culling = 2, flag_wireframe = 4, flag_occlusion = 8,
                   flag_prev_valid = 16, flag_software_raster = 32, flag_shadows = 64;
// Shadows are traced against a cut of each model with at most this many
// triangles: the finest cut that fits.
constexpr size_t shadow_triangle_budget = 1u << 18;
constexpr uint32_t max_hzb_levels = 16;

struct options {
    std::vector<std::string> models;
    int grid = 1;
    float spacing = 1.4f;
    int width = 1600, height = 900;
    bool headless = false, validate = false, vsync = true;
    int frames = 0;  // Headless: how many to draw (they are timed)
    std::string screenshot;
    float threshold = 1.0f;
    uint32_t mode = 0;
    bool camera_set = false;
    vec3 eye;
    float yaw = 0, pitch = 0;
    bool wireframe = false;
    uint32_t disable = 0;  // Flags turned off from the command line
    bool cull_only = false;
    float sw_pixels = 32;
};

struct camera {
    vec3 eye{0, 0.6f, 2.5f};
    float yaw = 0, pitch = -0.15f;  // Yaw 0 looks down -z
    float fov = 1.0f;               // Vertical, radians
    float near_z = 0.01f;

    vec3 forward() const {
        return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw)};
    }
    mat4 view_proj(float aspect) const {
        return perspective_reverse_z(fov, aspect, near_z) * look_to(eye, forward(), {0, 1, 0});
    }
};

// Frustum planes from a view-projection matrix (Gribb and Hartmann): left,
// right, bottom, top and near, normalized. There is no far plane.
void frustum_planes(const mat4& m, float out[5][4]) {
    auto row = [&](int r) { return std::array<float, 4>{m.at(r, 0), m.at(r, 1), m.at(r, 2), m.at(r, 3)}; };
    const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    const std::array<float, 4> planes[5] = {
        {r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]},
        {r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]},
        {r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]},
        {r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]},
        {r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]},  // Reversed depth: z <= w
    };
    for (int k = 0; k < 5; ++k) {
        const float l = std::sqrt(planes[k][0] * planes[k][0] + planes[k][1] * planes[k][1] + planes[k][2] * planes[k][2]);
        for (int c = 0; c < 4; ++c) out[k][c] = planes[k][c] / l;
    }
}

// The coarser copy of a model that shadow rays are traced against.
struct shadow_mesh {
    uint32_t first_vertex, vertex_count;  // In the scene's positions
    uint32_t first_index, index_count;    // In scene::shadow_indices, relative to first_vertex
};

struct scene {
    geometry all;  // Every model's arrays, concatenated
    std::vector<gpu_mesh> meshes;
    std::vector<gpu_instance> instances;
    std::vector<shadow_mesh> shadow_meshes;
    std::vector<uint32_t> shadow_indices;
    size_t instanced_triangles = 0;  // At full detail, over every instance
};

// The finest cut of a model's hierarchy within the shadow budget, as
// indices into its own vertices, and the error of that cut.
std::vector<uint32_t> shadow_cut(const geometry& g, float& error) {
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
    size_t lo = 0, hi = errors.size() - 1;  // The cut at the coarsest error is the roots: small enough
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (triangles_at(errors[mid]) <= shadow_triangle_budget) hi = mid;
        else lo = mid + 1;
    }
    error = errors[lo];
    std::vector<uint32_t> indices;
    for (const gpu_cluster& c : g.clusters) {
        if (!(c.lod_error <= error && error < c.parent_error)) continue;
        for (uint32_t t = 0; t < c.triangle_count; ++t) {
            const uint32_t p = g.cluster_triangles[c.triangle_offset + t];
            for (int k = 0; k < 3; ++k) indices.push_back(g.cluster_vertices[c.vertex_offset + ((p >> (8 * k)) & 255)]);
        }
    }
    return indices;
}

// Loads the models into one set of arrays, and places them on a grid,
// turned and sized at random.
scene make_scene(const options& opt) {
    scene s;
    std::vector<size_t> leaf_triangles;
    for (const std::string& path : opt.models) {
        geometry g = load_geometry(path);
        std::printf("%s: %zu clusters, %zu triangles at full detail, %zu levels\n", path.c_str(), g.clusters.size(),
                    g.leaf_triangles(), g.levels.size());
        leaf_triangles.push_back(g.leaf_triangles());
        float shadow_error = 0;
        const std::vector<uint32_t> shadow = shadow_cut(g, shadow_error);
        s.shadow_meshes.push_back({static_cast<uint32_t>(s.all.positions.size() / 3), static_cast<uint32_t>(g.positions.size() / 3),
                                   static_cast<uint32_t>(s.shadow_indices.size()), static_cast<uint32_t>(shadow.size())});
        s.shadow_indices.insert(s.shadow_indices.end(), shadow.begin(), shadow.end());
        gpu_mesh m{};
        m.shadow_error = shadow_error;
        m.first_cluster = static_cast<uint32_t>(s.all.clusters.size());
        m.cluster_count = static_cast<uint32_t>(g.clusters.size());
        m.bounds[0] = g.bounds.center.x; m.bounds[1] = g.bounds.center.y; m.bounds[2] = g.bounds.center.z;
        m.bounds[3] = g.bounds.radius;
        m.lod_bounds[0] = g.lod_bounds.center.x; m.lod_bounds[1] = g.lod_bounds.center.y;
        m.lod_bounds[2] = g.lod_bounds.center.z; m.lod_bounds[3] = g.lod_bounds.radius;
        s.meshes.push_back(m);
        const uint32_t vertex_base = static_cast<uint32_t>(s.all.positions.size() / 3);
        const uint32_t cv_base = static_cast<uint32_t>(s.all.cluster_vertices.size());
        const uint32_t ct_base = static_cast<uint32_t>(s.all.cluster_triangles.size());
        for (gpu_cluster c : g.clusters) {
            c.vertex_offset += cv_base;
            c.triangle_offset += ct_base;
            s.all.clusters.push_back(c);
        }
        for (uint32_t v : g.cluster_vertices) s.all.cluster_vertices.push_back(v + vertex_base);
        s.all.cluster_triangles.insert(s.all.cluster_triangles.end(), g.cluster_triangles.begin(), g.cluster_triangles.end());
        s.all.positions.insert(s.all.positions.end(), g.positions.begin(), g.positions.end());
        s.all.normals.insert(s.all.normals.end(), g.normals.begin(), g.normals.end());
    }

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> turn(0, 6.2831853f), size(0.85f, 1.15f);
    const int n = std::max(1, opt.grid);
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
            s.instances.push_back(inst);
            s.instanced_triangles += leaf_triangles[mesh];
        }
    return s;
}

std::string human(double v) {
    char b[32];
    if (v >= 1e9) std::snprintf(b, sizeof b, "%.2fB", v / 1e9);
    else if (v >= 1e6) std::snprintf(b, sizeof b, "%.2fM", v / 1e6);
    else if (v >= 1e3) std::snprintf(b, sizeof b, "%.1fk", v / 1e3);
    else std::snprintf(b, sizeof b, "%.0f", v);
    return b;
}

class renderer {
public:
    renderer(vk::context& ctx, const scene& sc, uint32_t width, uint32_t height, bool cull_only = false)
        : ctx_(ctx), sc_(sc), cull_only_(cull_only) {
        create_static_buffers();
        create_descriptors();
        create_pipelines();
        for (frame_slot& f : slots_) {
            f.cmd = ctx_.allocate_command_buffer();
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VK_CHECK(vkCreateFence(ctx_.device, &fci, nullptr, &f.fence));
            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VK_CHECK(vkCreateSemaphore(ctx_.device, &sci, nullptr, &f.image_ready));
            VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qci.queryCount = timestamp_count;
            VK_CHECK(vkCreateQueryPool(ctx_.device, &qci, nullptr, &f.queries));
            vkResetQueryPool(ctx_.device, f.queries, 0, timestamp_count);
        }
        resize(width, height);
    }

    ~renderer() {
        vkDeviceWaitIdle(ctx_.device);
        destroy_targets();
        for (frame_slot& f : slots_) {
            vkDestroyFence(ctx_.device, f.fence, nullptr);
            vkDestroySemaphore(ctx_.device, f.image_ready, nullptr);
            vkDestroyQueryPool(ctx_.device, f.queries, nullptr);
            ctx_.destroy(f.frame);
            ctx_.destroy(f.stats);
        }
        for (vk::buffer* b : {&clusters_, &cluster_vertices_, &cluster_triangles_, &positions_, &normals_, &meshes_,
                              &instances_, &work_, &visible_, &draw_args_, &readback_, &late_instances_, &late_clusters_})
            ctx_.destroy(*b);
        for (VkPipeline p : {instance_cull_, args_, shade_, raster_, hzb_, cluster_cull_, sw_raster_})
            vkDestroyPipeline(ctx_.device, p, nullptr);
        vkDestroySampler(ctx_.device, sampler_, nullptr);
        for (accel* a : {&tlas_}) {
            if (a->handle) ctx_.destroy_as(ctx_.device, a->handle, nullptr);
            ctx_.destroy(a->storage);
        }
        for (accel& a : blas_) {
            ctx_.destroy_as(ctx_.device, a.handle, nullptr);
            ctx_.destroy(a.storage);
        }
        ctx_.destroy(shadow_indices_);
        ctx_.destroy(as_instances_);
        vkDestroyPipelineLayout(ctx_.device, layout_, nullptr);
        vkDestroyDescriptorPool(ctx_.device, pool_, nullptr);
        vkDestroyDescriptorSetLayout(ctx_.device, set_layout_, nullptr);
    }

    // (Re)creates everything sized by the image.
    void resize(uint32_t width, uint32_t height) {
        vkDeviceWaitIdle(ctx_.device);
        destroy_targets();
        width_ = width;
        height_ = height;
        vis_ = ctx_.make_buffer(uint64_t(width) * height * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        depth_ = ctx_.make_image(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
        color_ = ctx_.make_image(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        for (frame_slot& f : slots_) {
            VkDescriptorBufferInfo vis_info{vis_.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo image_info{VK_NULL_HANDLE, color_.view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            w[0].dstSet = f.set;
            w[0].dstBinding = 10;
            w[0].descriptorCount = 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[0].pBufferInfo = &vis_info;
            w[1].dstSet = f.set;
            w[1].dstBinding = 13;
            w[1].descriptorCount = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo = &image_info;
            vkUpdateDescriptorSets(ctx_.device, 2, w, 0, nullptr);
        }
        create_hzb();
        prev_valid_ = false;
        ctx_.destroy(readback_);
        readback_ = ctx_.make_buffer(uint64_t(width) * height * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    }

    struct frame_result {
        gpu_stats stats{};
        // Pass 1 culling, pass 1 drawing, both pyramids and pass 2, shading,
        // and the total: from the frame that last used this slot.
        double ms[5] = {};
        bool valid = false;
    };

    // Records and submits one frame. With a swapchain image, the result is
    // blitted into it; without, it can be read back with read_pixels().
    // Returns what the GPU reported for the last frame drawn in this slot.
    frame_result draw(const gpu_frame& frame_in, VkImage target, VkSemaphore wait, VkSemaphore signal, bool readback) {
        frame_slot& f = slots_[slot_];
        slot_ = (slot_ + 1) % frames_in_flight;
        VK_CHECK(vkWaitForFences(ctx_.device, 1, &f.fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(ctx_.device, 1, &f.fence));

        frame_result result;
        if (f.used) {
            std::memcpy(&result.stats, f.stats.mapped, sizeof(gpu_stats));
            uint64_t ts[timestamp_count];
            if (vkGetQueryPoolResults(ctx_.device, f.queries, 0, timestamp_count, sizeof ts, ts, sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                const double tick = ctx_.properties.limits.timestampPeriod * 1e-6;
                for (int k = 0; k < 4; ++k) result.ms[k] = (ts[k + 1] - ts[k]) * tick;
                result.ms[4] = (ts[4] - ts[0]) * tick;
                result.valid = true;
            }
        }
        f.used = true;

        gpu_frame fr = frame_in;
        fr.width = width_;
        fr.height = height_;
        fr.instance_count = static_cast<uint32_t>(sc_.instances.size());
        fr.max_work_items = max_work_items;
        fr.max_visible = max_visible;
        fr.hzb_width = hzb_width_;
        fr.hzb_height = hzb_height_;
        fr.hzb_levels = hzb_levels_;
        std::memcpy(fr.prev_view, prev_view_, sizeof prev_view_);
        if (prev_valid_) fr.flags |= flag_prev_valid;
        std::memcpy(prev_view_, fr.view, sizeof prev_view_);
        prev_valid_ = true;
        std::memcpy(f.frame.mapped, &fr, sizeof fr);

        VkCommandBuffer cmd = f.cmd;
        VK_CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdResetQueryPool(cmd, f.queries, 0, timestamp_count);

        // The last frame's passes may still be reading what this one clears.
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 0);
        vkCmdFillBuffer(cmd, vis_.handle, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, work_.handle, 0, 16, 0);  // Both passes' counts and the late counts
        vkCmdFillBuffer(cmd, visible_.handle, 0, 16, 0);
        vkCmdFillBuffer(cmd, f.stats.handle, 0, sizeof(gpu_stats), 0);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &f.set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &f.set, 0, nullptr);
        vk::transition(cmd, depth_.handle, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        // Pass 1: what last frame's depth does not hide. Then the pyramid
        // from what it drew, and pass 2: what pass 1 thought hidden but is
        // not. Then the pyramid again, for next frame's pass 1.
        draw_pass(cmd, 0, f.queries);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 2);
        build_hzb(cmd);
        draw_pass(cmd, 1, f.queries);
        build_hzb(cmd);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 3);

        vk::transition(cmd, color_.handle, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shade_);
        vkCmdDispatch(cmd, (width_ + 7) / 8, (height_ + 7) / 8, 1);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 4);

        vk::transition(cmd, color_.handle, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        if (target) {
            vk::transition(cmd, target, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, 0,
                           VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {int32_t(width_), int32_t(height_), 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[1] = {int32_t(width_), int32_t(height_), 1};
            vkCmdBlitImage(cmd, color_.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
            vk::transition(cmd, target, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
        }
        if (readback) {
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {width_, height_, 1};
            vkCmdCopyImageToBuffer(cmd, color_.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_.handle, 1, &copy);
        }
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (wait) {
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &wait;
            si.pWaitDstStageMask = &wait_stage;
        }
        if (signal) {
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &signal;
        }
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(ctx_.queue, 1, &si, f.fence));
        return result;
    }

    // The next frame's acquire semaphore, once the frame that last used it
    // is done.
    VkSemaphore image_ready_semaphore() {
        VK_CHECK(vkWaitForFences(ctx_.device, 1, &slots_[slot_].fence, VK_TRUE, UINT64_MAX));
        return slots_[slot_].image_ready;
    }

    // The last frame drawn with readback, as 8-bit sRGB.
    std::vector<uint8_t> read_pixels() {
        vkDeviceWaitIdle(ctx_.device);
        std::vector<uint8_t> rgb(size_t(width_) * height_ * 3);
        const uint16_t* half = static_cast<const uint16_t*>(readback_.mapped);
        auto to_float = [](uint16_t h) {
            const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31, mant = h & 1023;
            float v = exp == 0 ? std::ldexp(float(mant), -24) : std::ldexp(float(mant | 1024), int(exp) - 25);
            return sign ? -v : v;
        };
        auto srgb = [](float c) {
            c = std::clamp(c, 0.0f, 1.0f);
            c = c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1 / 2.4f) - 0.055f;
            return uint8_t(std::lround(c * 255));
        };
        for (size_t i = 0; i < size_t(width_) * height_; ++i)
            for (int c = 0; c < 3; ++c) rgb[3 * i + c] = srgb(to_float(half[4 * i + c]));
        return rgb;
    }

private:
    static constexpr uint32_t timestamp_count = 5;

    struct frame_slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore image_ready = VK_NULL_HANDLE;
        VkQueryPool queries = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        vk::buffer frame, stats;
        bool used = false;
    };

    vk::context& ctx_;
    const scene& sc_;
    uint32_t width_ = 0, height_ = 0;
    std::array<frame_slot, frames_in_flight> slots_;
    uint32_t slot_ = 0;

    vk::buffer clusters_, cluster_vertices_, cluster_triangles_, positions_, normals_, meshes_, instances_;
    vk::buffer work_, visible_, draw_args_, vis_, readback_, late_instances_, late_clusters_;
    vk::image depth_, color_, hzb_image_;
    std::vector<VkImageView> hzb_views_;
    VkSampler sampler_ = VK_NULL_HANDLE;
    uint32_t hzb_width_ = 0, hzb_height_ = 0, hzb_levels_ = 0;
    float prev_view_[16] = {};
    bool prev_valid_ = false;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline instance_cull_ = VK_NULL_HANDLE, args_ = VK_NULL_HANDLE, shade_ = VK_NULL_HANDLE, raster_ = VK_NULL_HANDLE,
               hzb_ = VK_NULL_HANDLE, cluster_cull_ = VK_NULL_HANDLE, sw_raster_ = VK_NULL_HANDLE;
    bool cull_only_ = false;

    void destroy_targets() {
        ctx_.destroy(vis_);
        ctx_.destroy(depth_);
        ctx_.destroy(color_);
        for (VkImageView v : hzb_views_) vkDestroyImageView(ctx_.device, v, nullptr);
        hzb_views_.clear();
        ctx_.destroy(hzb_image_);
    }

    // The depth pyramid: level 0 half the screen (rounded up), each level
    // half the last, down to 1x1. One view of every level for sampling,
    // and one view per level for writing.
    void create_hzb() {
        hzb_width_ = (width_ + 1) / 2;
        hzb_height_ = (height_ + 1) / 2;
        hzb_levels_ = 1;
        while ((std::max(hzb_width_, hzb_height_) >> hzb_levels_) > 0 && hzb_levels_ < max_hzb_levels) ++hzb_levels_;
        VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ic.imageType = VK_IMAGE_TYPE_2D;
        ic.format = VK_FORMAT_R32_SFLOAT;
        ic.extent = {hzb_width_, hzb_height_, 1};
        ic.mipLevels = hzb_levels_;
        ic.arrayLayers = 1;
        ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling = VK_IMAGE_TILING_OPTIMAL;
        ic.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        hzb_image_ = ctx_.make_image(ic, VK_IMAGE_ASPECT_COLOR_BIT);
        for (uint32_t l = 0; l < hzb_levels_; ++l) {
            VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vc.image = hzb_image_.handle;
            vc.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vc.format = VK_FORMAT_R32_SFLOAT;
            vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, l, 1, 0, 1};
            VkImageView v;
            VK_CHECK(vkCreateImageView(ctx_.device, &vc, nullptr, &v));
            hzb_views_.push_back(v);
        }
        ctx_.submit([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            ib.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            ib.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib.image = hzb_image_.handle;
            ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, hzb_levels_, 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1;
            di.pImageMemoryBarriers = &ib;
            vkCmdPipelineBarrier2(cmd, &di);
        });
        for (frame_slot& f : slots_) {
            VkDescriptorImageInfo sampled{sampler_, hzb_image_.view, VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo levels[max_hzb_levels];
            for (uint32_t l = 0; l < max_hzb_levels; ++l)
                levels[l] = {VK_NULL_HANDLE, hzb_views_[std::min(l, hzb_levels_ - 1)], VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            w[0].dstSet = f.set;
            w[0].dstBinding = 16;
            w[0].descriptorCount = 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo = &sampled;
            w[1].dstSet = f.set;
            w[1].dstBinding = 17;
            w[1].descriptorCount = max_hzb_levels;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo = levels;
            vkUpdateDescriptorSets(ctx_.device, 2, w, 0, nullptr);
        }
    }

    void build_hzb(VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hzb_);
        for (uint32_t l = 0; l < hzb_levels_; ++l) {
            const gpu_push push{0, l};
            vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_ALL, 0, sizeof push, &push);
            vkCmdDispatch(cmd, ((hzb_width_ >> l) + 8) / 8, ((hzb_height_ >> l) + 8) / 8, 1);
            vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                        VK_ACCESS_2_SHADER_READ_BIT);
        }
    }

    // Runs args.comp for a pass: step 0 sizes the cluster culling, step 1
    // the draw.
    void write_args(VkCommandBuffer cmd, uint32_t pass, uint32_t step) {
        const gpu_push push{pass, step};
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_ALL, 0, sizeof push, &push);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, args_);
        vkCmdDispatch(cmd, 1, 1, 1);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        const gpu_push back{pass, 0};
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_ALL, 0, sizeof back, &back);
    }

    // Culls and draws one pass: instances, then clusters, into the
    // visibility buffer.
    void draw_pass(VkCommandBuffer cmd, uint32_t pass, VkQueryPool queries) {
        const gpu_push push{pass, 0};
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_ALL, 0, sizeof push, &push);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, instance_cull_);
        const uint32_t n = uint32_t(sc_.instances.size());
        vkCmdDispatch(cmd, std::min(n, 65535u), (n + 65534) / 65535, 1);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        write_args(cmd, pass, 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cluster_cull_);
        vkCmdDispatchIndirect(cmd, draw_args_.handle, 16 * pass);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        write_args(cmd, pass, 1);
        if (pass == 0) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, 1);

        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = depth_.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = pass == 0 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {0.0f, 0};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {width_, height_}};
        ri.layerCount = 1;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{0, 0, float(width_), float(height_), 0, 1};
        VkRect2D sc{{0, 0}, {width_, height_}};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, raster_);
        if (!cull_only_) ctx_.draw_mesh_tasks_indirect(cmd, draw_args_.handle, 32 + 16 * pass, 1, 16);
        vkCmdEndRendering(cmd);
        if (!cull_only_) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sw_raster_);
            vkCmdDispatchIndirect(cmd, draw_args_.handle, 64 + 16 * pass);
        }
        vk::barrier(cmd,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    }

    void create_static_buffers() {
        const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        const VkBufferUsageFlags as_input = ctx_.ray_query ? VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR : 0;
        clusters_ = ctx_.upload(sc_.all.clusters, ssbo);
        cluster_vertices_ = ctx_.upload(sc_.all.cluster_vertices, ssbo);
        cluster_triangles_ = ctx_.upload(sc_.all.cluster_triangles, ssbo);
        positions_ = ctx_.upload(sc_.all.positions, ssbo | as_input);
        if (ctx_.ray_query) build_shadow_scene();
        normals_ = ctx_.upload(sc_.all.normals, ssbo);
        meshes_ = ctx_.upload(sc_.meshes, ssbo);
        instances_ = ctx_.upload(sc_.instances, ssbo);
        work_ = ctx_.make_buffer(16 + 2 * uint64_t(max_work_items) * 8, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        visible_ = ctx_.make_buffer(16 + uint64_t(max_visible) * 8, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        draw_args_ = ctx_.make_buffer(128, ssbo | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false);
        late_instances_ = ctx_.make_buffer(4 * sc_.instances.size(), ssbo, false);
        late_clusters_ = ctx_.make_buffer(uint64_t(max_visible) * 8, ssbo, false);
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = VK_LOD_CLAMP_NONE;
        VK_CHECK(vkCreateSampler(ctx_.device, &sci, nullptr, &sampler_));
        for (frame_slot& f : slots_) {
            f.frame = ctx_.make_buffer(sizeof(gpu_frame), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
            f.stats = ctx_.make_buffer(sizeof(gpu_stats), ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        }
    }

    struct accel {
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        vk::buffer storage;
        VkDeviceAddress address = 0;
    };
    std::vector<accel> blas_;
    accel tlas_;
    vk::buffer shadow_indices_, as_instances_;

    // Builds an acceleration structure over one geometry and waits for it.
    accel build_accel(VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR& geom, uint32_t primitives) {
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type = type;
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.geometryCount = 1;
        info.pGeometries = &geom;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        ctx_.as_build_sizes(ctx_.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primitives, &sizes);
        accel a;
        a.storage = ctx_.make_buffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false);
        VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        ci.buffer = a.storage.handle;
        ci.size = sizes.accelerationStructureSize;
        ci.type = type;
        VK_CHECK(ctx_.create_as(ctx_.device, &ci, nullptr, &a.handle));
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &asp;
        vkGetPhysicalDeviceProperties2(ctx_.gpu, &p2);
        const VkDeviceSize align = asp.minAccelerationStructureScratchOffsetAlignment;
        vk::buffer scratch = ctx_.make_buffer(sizes.buildScratchSize + align, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
        info.dstAccelerationStructure = a.handle;
        info.scratchData.deviceAddress = (scratch.address + align - 1) / align * align;
        VkAccelerationStructureBuildRangeInfoKHR range{primitives, 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
        ctx_.submit([&](VkCommandBuffer cmd) { ctx_.build_as(cmd, 1, &info, &ranges); });
        ctx_.destroy(scratch);
        VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        ai.accelerationStructure = a.handle;
        a.address = ctx_.as_address(ctx_.device, &ai);
        return a;
    }

    // What shadow rays are traced against: per model, a bottom level over
    // the finest cut of its hierarchy within the shadow budget (see
    // shadow_cut()), and a top level over every instance. Two-sided, as
    // scans are drawn.
    void build_shadow_scene() {
        const auto start = std::chrono::steady_clock::now();
        shadow_indices_ = ctx_.upload(sc_.shadow_indices, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        size_t triangles = 0;
        for (const shadow_mesh& m : sc_.shadow_meshes) {
            VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            auto& tris = geom.geometry.triangles;
            tris.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            tris.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            tris.vertexData.deviceAddress = positions_.address + uint64_t(m.first_vertex) * 12;
            tris.vertexStride = 12;
            tris.maxVertex = m.vertex_count - 1;
            tris.indexType = VK_INDEX_TYPE_UINT32;
            tris.indexData.deviceAddress = shadow_indices_.address + uint64_t(m.first_index) * 4;
            blas_.push_back(build_accel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geom, m.index_count / 3));
            triangles += m.index_count / 3;
        }
        std::vector<VkAccelerationStructureInstanceKHR> instances(sc_.instances.size());
        for (size_t k = 0; k < instances.size(); ++k) {
            auto& ai = instances[k];
            std::memcpy(ai.transform.matrix, sc_.instances[k].rows, sizeof ai.transform.matrix);
            ai.instanceCustomIndex = static_cast<uint32_t>(k);
            ai.mask = 0xFF;
            ai.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            ai.accelerationStructureReference = blas_[sc_.instances[k].mesh].address;
        }
        as_instances_ = ctx_.upload(instances, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geom.geometry.instances.data.deviceAddress = as_instances_.address;
        tlas_ = build_accel(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, geom, static_cast<uint32_t>(instances.size()));
        std::printf("shadows: %s triangles over %zu models, built in %.2f s\n", human(double(triangles)).c_str(),
                    sc_.shadow_meshes.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }

    void create_descriptors() {
        std::vector<VkDescriptorSetLayoutBinding> b;
        const uint32_t last_binding = ctx_.ray_query ? 18 : 17;
        for (uint32_t i = 0; i <= last_binding; ++i) {
            VkDescriptorSetLayoutBinding x{};
            x.binding = i;
            x.descriptorCount = i == 17 ? max_hzb_levels : 1;
            x.descriptorType = i == 0                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                             : i == 13 || i == 17    ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                             : i == 16               ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                             : i == 18               ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                                                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            x.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
            b.push_back(x);
        }
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = static_cast<uint32_t>(b.size());
        lci.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &set_layout_));

        const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 14 * frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, (1 + max_hzb_levels) * frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, frames_in_flight}};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = frames_in_flight;
        pci.poolSizeCount = ctx_.ray_query ? 5 : 4;
        pci.pPoolSizes = sizes;
        VK_CHECK(vkCreateDescriptorPool(ctx_.device, &pci, nullptr, &pool_));

        for (frame_slot& f : slots_) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = pool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &set_layout_;
            VK_CHECK(vkAllocateDescriptorSets(ctx_.device, &ai, &f.set));
            const vk::buffer* buffers[] = {&f.frame,   &clusters_, &cluster_vertices_, &cluster_triangles_, &positions_,
                                           &normals_,  &meshes_,   &instances_,        &work_,              &visible_,
                                           nullptr,    &f.stats,   &draw_args_,        nullptr,             &late_instances_,
                                           &late_clusters_};
            VkDescriptorBufferInfo infos[16];
            std::vector<VkWriteDescriptorSet> writes;
            for (uint32_t i = 0; i < 16; ++i) {
                if (!buffers[i]) continue;  // The visibility buffer and output image: written by resize()
                infos[i] = {buffers[i]->handle, 0, VK_WHOLE_SIZE};
                VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.dstSet = f.set;
                w.dstBinding = i;
                w.descriptorCount = 1;
                w.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w.pBufferInfo = &infos[i];
                writes.push_back(w);
            }
            VkWriteDescriptorSetAccelerationStructureKHR as_info{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            as_info.accelerationStructureCount = 1;
            as_info.pAccelerationStructures = &tlas_.handle;
            if (ctx_.ray_query) {
                VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.pNext = &as_info;
                w.dstSet = f.set;
                w.dstBinding = 18;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                writes.push_back(w);
            }
            vkUpdateDescriptorSets(ctx_.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        const VkPushConstantRange push{VK_SHADER_STAGE_ALL, 0, sizeof(gpu_push)};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &set_layout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(ctx_.device, &plci, nullptr, &layout_));
    }

    VkPipeline compute_pipeline(const uint32_t* code, size_t bytes) {
        VkShaderModule m = ctx_.shader(code, bytes);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = m;
        ci.stage.pName = "main";
        ci.layout = layout_;
        VkPipeline p;
        VK_CHECK(vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
        vkDestroyShaderModule(ctx_.device, m, nullptr);
        return p;
    }

    void create_pipelines() {
        instance_cull_ = compute_pipeline(instance_cull_spv, sizeof instance_cull_spv);
        args_ = compute_pipeline(args_spv, sizeof args_spv);
        shade_ = ctx_.ray_query ? compute_pipeline(shade_rt_spv, sizeof shade_rt_spv) : compute_pipeline(shade_spv, sizeof shade_spv);
        cluster_cull_ = compute_pipeline(cluster_cull_spv, sizeof cluster_cull_spv);
        sw_raster_ = compute_pipeline(sw_raster_spv, sizeof sw_raster_spv);
        hzb_ = compute_pipeline(hzb_spv, sizeof hzb_spv);

        VkShaderModule mesh = ctx_.shader(draw_mesh_spv, sizeof draw_mesh_spv);
        VkShaderModule frag = ctx_.shader(vis_frag_spv, sizeof vis_frag_spv);
        VkPipelineShaderStageCreateInfo stages[2] = {};
        const VkShaderStageFlagBits kinds[2] = {VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT};
        const VkShaderModule modules[2] = {mesh, frag};
        for (int i = 0; i < 2; ++i) {
            stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[i].stage = kinds[i];
            stages[i].module = modules[i];
            stages[i].pName = "main";
        }
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        // Scans have holes, through which the inside shows: draw both sides.
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;  // Reversed depth
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
        VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gci.pNext = &rendering;
        gci.stageCount = 2;
        gci.pStages = stages;
        gci.pViewportState = &viewport;
        gci.pRasterizationState = &raster;
        gci.pMultisampleState = &ms;
        gci.pDepthStencilState = &ds;
        gci.pColorBlendState = &blend;
        gci.pDynamicState = &dynamic;
        gci.layout = layout_;
        VK_CHECK(vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &gci, nullptr, &raster_));
        for (VkShaderModule m : modules) vkDestroyShaderModule(ctx_.device, m, nullptr);
    }
};

class swapchain {
public:
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    std::vector<VkImage> images;
    std::vector<VkSemaphore> done;  // One per image: presenting waits on it
    VkExtent2D extent{};

    swapchain(vk::context& ctx, bool vsync) : ctx_(ctx), vsync_(vsync) {}
    ~swapchain() { destroy(); }

    void create(uint32_t width, uint32_t height) {
        VkSurfaceCapabilitiesKHR caps;
        VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.gpu, ctx_.surface, &caps));
        extent = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{width, height};
        uint32_t n = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.gpu, ctx_.surface, &n, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(n);
        vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.gpu, ctx_.surface, &n, formats.data());
        VkSurfaceFormatKHR format = formats[0];
        for (const auto& f : formats)
            if ((f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_R8G8B8A8_SRGB) &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                format = f;
        vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.gpu, ctx_.surface, &n, nullptr);
        std::vector<VkPresentModeKHR> modes(n);
        vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.gpu, ctx_.surface, &n, modes.data());
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        if (!vsync_)
            for (VkPresentModeKHR m : modes)
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR || (m == VK_PRESENT_MODE_MAILBOX_KHR && mode != VK_PRESENT_MODE_IMMEDIATE_KHR))
                    mode = m;

        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface = ctx_.surface;
        sci.minImageCount = std::max(caps.minImageCount, 3u);
        if (caps.maxImageCount) sci.minImageCount = std::min(sci.minImageCount, caps.maxImageCount);
        sci.imageFormat = format.format;
        sci.imageColorSpace = format.colorSpace;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = mode;
        sci.clipped = VK_TRUE;
        sci.oldSwapchain = handle;
        VkSwapchainKHR fresh;
        VK_CHECK(vkCreateSwapchainKHR(ctx_.device, &sci, nullptr, &fresh));
        destroy();
        handle = fresh;
        vkGetSwapchainImagesKHR(ctx_.device, handle, &n, nullptr);
        images.resize(n);
        vkGetSwapchainImagesKHR(ctx_.device, handle, &n, images.data());
        done.resize(n);
        for (VkSemaphore& s : done) {
            VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VK_CHECK(vkCreateSemaphore(ctx_.device, &ci, nullptr, &s));
        }
    }

private:
    vk::context& ctx_;
    bool vsync_;

    void destroy() {
        if (!handle) return;
        vkDeviceWaitIdle(ctx_.device);
        for (VkSemaphore s : done) vkDestroySemaphore(ctx_.device, s, nullptr);
        done.clear();
        vkDestroySwapchainKHR(ctx_.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
};

struct view_state {
    camera cam;
    camera cull_cam;  // Equal to cam unless frozen
    bool frozen = false;
    float threshold = 1;
    uint32_t mode = 0;
    uint32_t flags = flag_cone_culling | flag_frustum_culling | flag_occlusion | flag_software_raster | flag_shadows;
    float sw_max_pixels = 32;
    float speed = 1.5f;
    bool looking = false;
    double last_x = 0, last_y = 0;
    bool print_camera = false;
};

gpu_frame frame_data(const view_state& v, uint32_t width, uint32_t height, float time) {
    gpu_frame f{};
    const float aspect = float(width) / float(height);
    const mat4 vp = v.cam.view_proj(aspect);
    std::memcpy(f.view_proj, vp.m, sizeof f.view_proj);
    const mat4 inv = inverse(vp);
    std::memcpy(f.inv_view_proj, inv.m, sizeof f.inv_view_proj);
    const camera& cc = v.frozen ? v.cull_cam : v.cam;
    frustum_planes(cc.view_proj(aspect), f.cull_planes);
    f.cull_origin[0] = cc.eye.x; f.cull_origin[1] = cc.eye.y; f.cull_origin[2] = cc.eye.z;
    f.origin[0] = v.cam.eye.x; f.origin[1] = v.cam.eye.y; f.origin[2] = v.cam.eye.z;
    const mat4 view = look_to(v.cam.eye, v.cam.forward(), {0, 1, 0});
    std::memcpy(f.view, view.m, sizeof f.view);
    const mat4 proj = perspective_reverse_z(v.cam.fov, aspect, v.cam.near_z);
    f.p00 = proj.at(0, 0);
    f.p11 = -proj.at(1, 1);  // The projection flips y for Vulkan; the sphere test wants it upright
    f.flags = v.flags;
    // Depth from the drawing camera says nothing about what a frozen
    // culling camera would see.
    if (v.frozen) f.flags &= ~flag_occlusion;
    f.lod_scale = float(height) / (2 * std::tan(v.cam.fov / 2));
    f.lod_threshold = v.threshold;
    f.sw_max_pixels = v.sw_max_pixels;
    f.near_z = v.cam.near_z;
    f.debug_mode = v.mode;
    f.time = time;
    return f;
}

void on_key(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* v = static_cast<view_state*>(glfwGetWindowUserPointer(w));
    if (key >= GLFW_KEY_1 && key < GLFW_KEY_1 + int(mode_count)) v->mode = uint32_t(key - GLFW_KEY_1);
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, 1);
    if (key == GLFW_KEY_LEFT_BRACKET) v->threshold = std::max(0.125f, v->threshold / 2);
    if (key == GLFW_KEY_RIGHT_BRACKET) v->threshold = std::min(256.0f, v->threshold * 2);
    if (key == GLFW_KEY_C) v->flags ^= flag_cone_culling;
    if (key == GLFW_KEY_V) v->flags ^= flag_frustum_culling;
    if (key == GLFW_KEY_O) v->flags ^= flag_occlusion;
    if (key == GLFW_KEY_R) v->flags ^= flag_software_raster;
    if (key == GLFW_KEY_H) v->flags ^= flag_shadows;
    if (key == GLFW_KEY_T) v->flags ^= flag_wireframe;
    if (key == GLFW_KEY_P) v->print_camera = true;
    if (key == GLFW_KEY_F) {
        v->frozen = !v->frozen;
        v->cull_cam = v->cam;
    }
}

void on_scroll(GLFWwindow* w, double, double dy) {
    auto* v = static_cast<view_state*>(glfwGetWindowUserPointer(w));
    v->speed = std::clamp(v->speed * float(std::pow(1.25, dy)), 0.05f, 500.0f);
}

void move_camera(GLFWwindow* w, view_state& v, float dt) {
    double x, y;
    glfwGetCursorPos(w, &x, &y);
    const bool look = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
                      glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (look && v.looking) {
        v.cam.yaw += float(x - v.last_x) * 0.003f;
        v.cam.pitch = std::clamp(v.cam.pitch - float(y - v.last_y) * 0.003f, -1.55f, 1.55f);
    }
    v.looking = look;
    v.last_x = x;
    v.last_y = y;
    const vec3 f = v.cam.forward(), r = normalize(cross(f, {0, 1, 0}));
    vec3 move(0, 0, 0);
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) move += f;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) move += -f;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) move += r;
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) move += -r;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) move += vec3(0, 1, 0);
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) move += vec3(0, -1, 0);
    const float boost = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 4.0f : 1.0f;
    v.cam.eye += move * (v.speed * boost * dt);
}

std::string stats_line(const renderer::frame_result& r, const view_state& v, size_t instances) {
    char b[400];
    std::snprintf(b, sizeof b, "%.2f ms (cull %.2f, raster %.2f, pass 2 %.2f, shade %.2f) | %s tris, %s clusters (%s software, %s late) | %u/%zu instances | %.3gpx | %s%s%s%s%s%s",
                  r.ms[4], r.ms[0], r.ms[1], r.ms[2], r.ms[3], human(r.stats.triangles_drawn).c_str(),
                  human(r.stats.clusters_drawn).c_str(), human(r.stats.clusters_software).c_str(), human(r.stats.clusters_late).c_str(), r.stats.instances_visible,
                  instances, v.threshold,
                  mode_names[v.mode], v.frozen ? " | FROZEN" : "", (v.flags & flag_cone_culling) ? "" : " | no cone",
                  (v.flags & flag_occlusion) ? "" : " | no occlusion", (v.flags & flag_software_raster) ? "" : " | no software raster",
                  (r.stats.work_overflow || r.stats.visible_overflow) ? " | OVERFLOW" : "");
    return b;
}

options parse(int argc, char** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") o.models.push_back(next());
        else if (a == "--grid") o.grid = std::stoi(next());
        else if (a == "--spacing") o.spacing = std::stof(next());
        else if (a == "--width") o.width = std::stoi(next());
        else if (a == "--height") o.height = std::stoi(next());
        else if (a == "--headless") o.headless = true;
        else if (a == "--frames") o.frames = std::stoi(next());
        else if (a == "--screenshot") o.screenshot = next();
        else if (a == "--threshold") o.threshold = std::stof(next());
        else if (a == "--mode") o.mode = std::min<uint32_t>(std::stoul(next()), mode_count - 1);
        else if (a == "--validate") o.validate = true;
        else if (a == "--no-vsync") o.vsync = false;
        else if (a == "--wireframe") o.wireframe = true;
        else if (a == "--no-occlusion") o.disable |= flag_occlusion;
        else if (a == "--no-cone") o.disable |= flag_cone_culling;
        else if (a == "--no-sw") o.disable |= flag_software_raster;
        else if (a == "--no-shadows") o.disable |= flag_shadows;
        else if (a == "--sw-pixels") o.sw_pixels = std::clamp(std::stof(next()), 0.0f, 64.0f);  // sw_raster.comp's 32-bit math holds to 64
        else if (a == "--cull-only") o.cull_only = true;
        else if (a == "--camera") {
            float x, y, z, yaw, pitch;
            if (std::sscanf(next().c_str(), "%f,%f,%f,%f,%f", &x, &y, &z, &yaw, &pitch) != 5)
                throw std::runtime_error("--camera wants x,y,z,yaw,pitch");
            o.eye = {x, y, z};
            o.yaw = yaw;
            o.pitch = pitch;
            o.camera_set = true;
        } else throw std::runtime_error("unknown argument " + a);
    }
    if (o.models.empty()) throw std::runtime_error("usage: nexus_view --model FILE.ngeo [--model ...] [--grid N] [--headless --frames N --screenshot out.png]");
    if (o.headless && o.frames <= 0) o.frames = 1;
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const options opt = parse(argc, argv);
        const scene sc = make_scene(opt);
        std::printf("%zu instances, %s triangles at full detail\n", sc.instances.size(), human(double(sc.instanced_triangles)).c_str());

        view_state v;
        v.threshold = opt.threshold;
        v.mode = opt.mode;
        if (opt.wireframe) v.flags |= flag_wireframe;
        v.flags &= ~opt.disable;
        v.sw_max_pixels = opt.sw_pixels;
        const float extent = std::max(1.0f, opt.grid * opt.spacing);
        v.cam.eye = {0, 0.35f + 0.25f * extent, 0.5f + 0.75f * extent};
        v.cam.pitch = opt.grid > 1 ? -0.35f : -0.1f;
        if (opt.camera_set) {
            v.cam.eye = opt.eye;
            v.cam.yaw = opt.yaw;
            v.cam.pitch = opt.pitch;
        }
        v.cull_cam = v.cam;

        if (opt.headless) {
            vk::context ctx(nullptr, opt.validate);
            std::printf("GPU: %s\n", ctx.device_name.c_str());
            renderer r(ctx, sc, opt.width, opt.height, opt.cull_only);
            std::vector<renderer::frame_result> results;
            for (int i = 0; i < opt.frames + 2; ++i) {
                const bool last = i == opt.frames - 1;
                auto res = r.draw(frame_data(v, opt.width, opt.height, 0), VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                  last && !opt.screenshot.empty());
                if (res.valid) results.push_back(res);
                if (last && !opt.screenshot.empty()) {
                    png::write_rgb(opt.screenshot, opt.width, opt.height, r.read_pixels());
                    std::printf("wrote %s\n", opt.screenshot.c_str());
                }
            }
            vkDeviceWaitIdle(ctx.device);
            if (!results.empty()) {
                // The median frame, by total time.
                std::sort(results.begin(), results.end(), [](auto& a, auto& b) { return a.ms[4] < b.ms[4]; });
                std::printf("%s\n", stats_line(results[results.size() / 2], v, sc.instances.size()).c_str());
                const auto& s = results[results.size() / 2].stats;
                std::printf("clusters tested %s, work items %s, hidden by last frame %s, instances hidden %u\n",
                            human(s.clusters_tested).c_str(), human(s.work_items).c_str(), human(s.clusters_occluded).c_str(),
                            s.instances_occluded);
            }
            return vk::validation_errors ? 1 : 0;
        }

        if (!glfwInit()) throw std::runtime_error("glfwInit failed");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(opt.width, opt.height, "nexus_geometry", nullptr, nullptr);
        if (!window) throw std::runtime_error("cannot open a window");
        glfwSetWindowUserPointer(window, &v);
        glfwSetKeyCallback(window, on_key);
        glfwSetScrollCallback(window, on_scroll);
        {
            vk::context ctx(window, opt.validate);
            std::printf("GPU: %s\n", ctx.device_name.c_str());
            int fw, fh;
            glfwGetFramebufferSize(window, &fw, &fh);
            swapchain swap(ctx, opt.vsync);
            swap.create(fw, fh);
            renderer r(ctx, sc, swap.extent.width, swap.extent.height);
            std::printf("keys: WASD/QE move, drag to look, scroll for speed, shift to hurry\n"
                        "      1-8 view (shaded, clusters, triangles, LOD level, groups, instances, holes, rasterizer)\n"
                        "      [ ] LOD threshold, F freeze culling, C cone, V frustum, O occlusion culling,\n"
                        "      R software rasterizer, H shadows, T wireframe, P print camera\n");

            auto last = std::chrono::steady_clock::now();
            const auto start = last;
            double title_timer = 0;
            bool rebuild = false;
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                const auto now = std::chrono::steady_clock::now();
                const float dt = std::chrono::duration<float>(now - last).count();
                last = now;
                move_camera(window, v, dt);
                if (v.print_camera) {
                    std::printf("--camera %.3f,%.3f,%.3f,%.3f,%.3f\n", v.cam.eye.x, v.cam.eye.y, v.cam.eye.z, v.cam.yaw, v.cam.pitch);
                    v.print_camera = false;
                }

                glfwGetFramebufferSize(window, &fw, &fh);
                if (fw == 0 || fh == 0) {
                    glfwWaitEvents();
                    continue;
                }
                if (rebuild || uint32_t(fw) != swap.extent.width || uint32_t(fh) != swap.extent.height) {
                    swap.create(fw, fh);
                    r.resize(swap.extent.width, swap.extent.height);
                    rebuild = false;
                }
                VkSemaphore ready = r.image_ready_semaphore();
                uint32_t index;
                VkResult acq = vkAcquireNextImageKHR(ctx.device, swap.handle, UINT64_MAX, ready, VK_NULL_HANDLE, &index);
                if (acq == VK_ERROR_OUT_OF_DATE_KHR) { rebuild = true; continue; }
                if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) VK_CHECK(acq);
                const float time = std::chrono::duration<float>(now - start).count();
                auto res = r.draw(frame_data(v, swap.extent.width, swap.extent.height, time), swap.images[index], ready,
                                  swap.done[index], false);
                VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
                pi.waitSemaphoreCount = 1;
                pi.pWaitSemaphores = &swap.done[index];
                pi.swapchainCount = 1;
                pi.pSwapchains = &swap.handle;
                pi.pImageIndices = &index;
                const VkResult pr = vkQueuePresentKHR(ctx.queue, &pi);
                if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) rebuild = true;
                else VK_CHECK(pr);

                title_timer += dt;
                if (res.valid && title_timer > 0.25) {
                    title_timer = 0;
                    glfwSetWindowTitle(window, ("nexus_geometry | " + stats_line(res, v, sc.instances.size())).c_str());
                }
            }
            vkDeviceWaitIdle(ctx.device);
        }
        glfwDestroyWindow(window);
        glfwTerminate();
        return vk::validation_errors ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
