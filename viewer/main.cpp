// nexus_geometry viewer: draws instanced cluster LOD hierarchies (.ngeo
// files from ngeo_build) with task and mesh shaders on Vulkan.
//
// Each frame:
//   1. instance_cull.comp drops instances outside the view and, for each
//      one left, finds by binary search the clusters that might be drawn
//      at its distance, as work items of 64 clusters.
//   2. cull.task tests each of those clusters: the LOD cut, the frustum and
//      the normal cone. Survivors are numbered in a visible list.
//   3. draw.mesh draws each survivor; vis.frag writes depth and triangle
//      into a 64-bit visibility buffer with an atomic max.
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
const uint32_t cull_task_spv[] = {
#include "cull.task.inc"
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

constexpr uint32_t frames_in_flight = 2;
constexpr uint32_t max_work_items = 1u << 22;
constexpr uint32_t max_visible = 1u << 22;  // Leaves 7 bits for the triangle in a 32-bit id... and 3 to spare
constexpr const char* mode_names[] = {"shaded", "clusters", "triangles", "LOD level", "groups", "instances", "holes"};
constexpr uint32_t mode_count = 7;

// Laid out as viewer/shaders/common.glsl declares them (scalar layout).
struct gpu_mesh {
    uint32_t first_cluster, cluster_count, pad0, pad1;
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
};
struct gpu_stats {
    uint32_t instances_visible, work_items, clusters_tested, clusters_drawn, triangles_drawn;
    uint32_t work_overflow, visible_overflow, pad;
};

constexpr uint32_t flag_cone_culling = 1, flag_frustum_culling = 2, flag_wireframe = 4;

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

struct scene {
    geometry all;  // Every model's arrays, concatenated
    std::vector<gpu_mesh> meshes;
    std::vector<gpu_instance> instances;
    size_t instanced_triangles = 0;  // At full detail, over every instance
};

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
        gpu_mesh m{};
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
    renderer(vk::context& ctx, const scene& sc, uint32_t width, uint32_t height) : ctx_(ctx), sc_(sc) {
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
                              &instances_, &work_, &visible_, &draw_args_, &readback_})
            ctx_.destroy(*b);
        for (VkPipeline p : {instance_cull_, args_, shade_, raster_}) vkDestroyPipeline(ctx_.device, p, nullptr);
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
        ctx_.destroy(readback_);
        readback_ = ctx_.make_buffer(uint64_t(width) * height * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    }

    struct frame_result {
        gpu_stats stats{};
        double ms[4] = {};  // Cull, raster, shade, total: from the frame that used this slot last
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
                result.ms[0] = (ts[1] - ts[0]) * tick;
                result.ms[1] = (ts[2] - ts[1]) * tick;
                result.ms[2] = (ts[3] - ts[2]) * tick;
                result.ms[3] = (ts[3] - ts[0]) * tick;
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
        vkCmdFillBuffer(cmd, work_.handle, 0, 16, 0);
        vkCmdFillBuffer(cmd, visible_.handle, 0, 16, 0);
        vkCmdFillBuffer(cmd, f.stats.handle, 0, sizeof(gpu_stats), 0);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &f.set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &f.set, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, instance_cull_);
        vkCmdDispatch(cmd, (fr.instance_count + 63) / 64, 1, 1);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, args_);
        vkCmdDispatch(cmd, 1, 1, 1);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
                    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 1);

        vk::transition(cmd, depth_.handle, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = depth_.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
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
        ctx_.draw_mesh_tasks_indirect(cmd, draw_args_.handle, 0, 1, 12);
        vkCmdEndRendering(cmd);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 2);

        vk::barrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
                    VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        vk::transition(cmd, color_.handle, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shade_);
        vkCmdDispatch(cmd, (width_ + 7) / 8, (height_ + 7) / 8, 1);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 3);

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
    static constexpr uint32_t timestamp_count = 4;

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
    vk::buffer work_, visible_, draw_args_, vis_, readback_;
    vk::image depth_, color_;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline instance_cull_ = VK_NULL_HANDLE, args_ = VK_NULL_HANDLE, shade_ = VK_NULL_HANDLE, raster_ = VK_NULL_HANDLE;

    void destroy_targets() {
        ctx_.destroy(vis_);
        ctx_.destroy(depth_);
        ctx_.destroy(color_);
    }

    void create_static_buffers() {
        const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        clusters_ = ctx_.upload(sc_.all.clusters, ssbo);
        cluster_vertices_ = ctx_.upload(sc_.all.cluster_vertices, ssbo);
        cluster_triangles_ = ctx_.upload(sc_.all.cluster_triangles, ssbo);
        positions_ = ctx_.upload(sc_.all.positions, ssbo);
        normals_ = ctx_.upload(sc_.all.normals, ssbo);
        meshes_ = ctx_.upload(sc_.meshes, ssbo);
        instances_ = ctx_.upload(sc_.instances, ssbo);
        work_ = ctx_.make_buffer(16 + uint64_t(max_work_items) * 8, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        visible_ = ctx_.make_buffer(16 + uint64_t(max_visible) * 8, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        draw_args_ = ctx_.make_buffer(16, ssbo | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false);
        for (frame_slot& f : slots_) {
            f.frame = ctx_.make_buffer(sizeof(gpu_frame), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
            f.stats = ctx_.make_buffer(sizeof(gpu_stats), ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        }
    }

    void create_descriptors() {
        std::vector<VkDescriptorSetLayoutBinding> b;
        for (uint32_t i = 0; i <= 13; ++i) {
            VkDescriptorSetLayoutBinding x{};
            x.binding = i;
            x.descriptorCount = 1;
            x.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                             : i == 13 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            x.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                           VK_SHADER_STAGE_FRAGMENT_BIT;
            b.push_back(x);
        }
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = static_cast<uint32_t>(b.size());
        lci.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &set_layout_));

        const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12 * frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, frames_in_flight}};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = frames_in_flight;
        pci.poolSizeCount = 3;
        pci.pPoolSizes = sizes;
        VK_CHECK(vkCreateDescriptorPool(ctx_.device, &pci, nullptr, &pool_));

        for (frame_slot& f : slots_) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = pool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &set_layout_;
            VK_CHECK(vkAllocateDescriptorSets(ctx_.device, &ai, &f.set));
            const vk::buffer* buffers[] = {&f.frame, &clusters_, &cluster_vertices_, &cluster_triangles_, &positions_,
                                           &normals_, &meshes_, &instances_, &work_, &visible_, nullptr, &f.stats, &draw_args_};
            VkDescriptorBufferInfo infos[13];
            std::vector<VkWriteDescriptorSet> writes;
            for (uint32_t i = 0; i < 13; ++i) {
                if (!buffers[i]) continue;  // The visibility buffer: written by resize()
                infos[i] = {buffers[i]->handle, 0, VK_WHOLE_SIZE};
                VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.dstSet = f.set;
                w.dstBinding = i;
                w.descriptorCount = 1;
                w.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w.pBufferInfo = &infos[i];
                writes.push_back(w);
            }
            vkUpdateDescriptorSets(ctx_.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &set_layout_;
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
        shade_ = compute_pipeline(shade_spv, sizeof shade_spv);

        VkShaderModule task = ctx_.shader(cull_task_spv, sizeof cull_task_spv);
        VkShaderModule mesh = ctx_.shader(draw_mesh_spv, sizeof draw_mesh_spv);
        VkShaderModule frag = ctx_.shader(vis_frag_spv, sizeof vis_frag_spv);
        VkPipelineShaderStageCreateInfo stages[3] = {};
        const VkShaderStageFlagBits kinds[3] = {VK_SHADER_STAGE_TASK_BIT_EXT, VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT};
        const VkShaderModule modules[3] = {task, mesh, frag};
        for (int i = 0; i < 3; ++i) {
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
        gci.stageCount = 3;
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
    uint32_t flags = flag_cone_culling | flag_frustum_culling;
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
    f.flags = v.flags;
    f.lod_scale = float(height) / (2 * std::tan(v.cam.fov / 2));
    f.lod_threshold = v.threshold;
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
    std::snprintf(b, sizeof b, "%.2f ms (cull %.2f, raster %.2f, shade %.2f) | %s tris, %s clusters | %u/%zu instances | %.3gpx | %s%s%s%s",
                  r.ms[3], r.ms[0], r.ms[1], r.ms[2], human(r.stats.triangles_drawn).c_str(),
                  human(r.stats.clusters_drawn).c_str(), r.stats.instances_visible, instances, v.threshold,
                  mode_names[v.mode], v.frozen ? " | FROZEN" : "", (v.flags & flag_cone_culling) ? "" : " | no cone",
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
            renderer r(ctx, sc, opt.width, opt.height);
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
                std::sort(results.begin(), results.end(), [](auto& a, auto& b) { return a.ms[3] < b.ms[3]; });
                std::printf("%s\n", stats_line(results[results.size() / 2], v, sc.instances.size()).c_str());
                const auto& s = results[results.size() / 2].stats;
                std::printf("clusters tested %s, work items %s\n", human(s.clusters_tested).c_str(), human(s.work_items).c_str());
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
                        "      1-7 view (shaded, clusters, triangles, LOD level, groups, instances, holes)\n"
                        "      [ ] LOD threshold, F freeze culling, C cone culling, V frustum culling, T wireframe, P print camera\n");

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
