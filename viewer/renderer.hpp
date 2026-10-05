#pragma once
// the renderer: vulkan resources, pipelines and each frame's passes (see main.cpp's outline).
#include "camera.hpp"
#include "gpu_types.hpp"
#include "scene.hpp"
#include "shaders.hpp"
#include "streamer.hpp"
#include "vk.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace viewer {

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
    renderer(vk::context& ctx, const scene& sc, uint32_t width, uint32_t height, uint64_t pool_bytes, uint64_t upload_bytes,
             unsigned loader_threads, bool prefetch, bool cull_only = false, uint32_t vsm_side = 32, bool merge_reads = true)
        : ctx_(ctx), sc_(sc), vsm_side_(vsm_side), cull_only_(cull_only), upload_bytes_(upload_bytes),
          streamer_(sc.pages, sc.deps, pool_bytes, loader_threads, prefetch ? sc.children : std::vector<uint32_t>{}, merge_reads) {
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
            for (vk::buffer* b : {&f.pose_now, &f.pose_prev, &f.pose_info}) ctx_.destroy(*b);
        }
        for (frame_slot& f : slots_)
            for (vk::buffer* b : {&f.staging, &f.request_readback, &f.used_readback}) ctx_.destroy(*b);
        for (vk::buffer* b : {&clusters_, &page_table_, &pool_buffer_, &page_used_, &requests_, &request_stamp_, &shadow_positions_, &meshes_,
                              &instances_, &work_, &visible_, &draw_args_, &readback_, &late_instances_, &late_clusters_,
                              &cells_, &cell_lists_, &vsm_entries_, &vsm_phys_, &vsm_lists_, &vsm_atlas_, &vsm_work_,
                              &vsm_visible_, &vsm_args_, &moving_instances_, &page_skins_})
            ctx_.destroy(*b);
        vkDestroySampler(ctx_.device, history_sampler_, nullptr);
        for (VkPipeline p : {instance_cull_, args_, shade_, raster_, hzb_, cluster_cull_, sw_raster_, shadow_, taa_, expand_, tlas_update_, cell_cull_,
                            vsm_mark_, vsm_alloc_, vsm_clear_, vsm_args_pipeline_, vsm_instance_, vsm_expand_, vsm_cluster_,
                            vsm_raster_, ao_pipeline_, ao_rt_pipeline_, ao_depth_pipeline_})
            vkDestroyPipeline(ctx_.device, p, nullptr);
        vkDestroySampler(ctx_.device, sampler_, nullptr);
        for (accel* a : {&tlas_, &tlas_moving_}) {
            if (a->handle) ctx_.destroy_as(ctx_.device, a->handle, nullptr);
            ctx_.destroy(a->storage);
        }
        for (accel& a : blas_) {
            ctx_.destroy_as(ctx_.device, a.handle, nullptr);
            ctx_.destroy(a.storage);
        }
        ctx_.destroy(shadow_indices_);
        ctx_.destroy(as_instances_);
        ctx_.destroy(as_moving_);
        ctx_.destroy(tlas_scratch_);
        vkDestroyPipelineLayout(ctx_.device, layout_, nullptr);
        vkDestroyDescriptorPool(ctx_.device, pool_, nullptr);
        vkDestroyDescriptorSetLayout(ctx_.device, set_layout_, nullptr);
    }

    // (re)creates everything sized by the image.
    void resize(uint32_t width, uint32_t height) {
        vkDeviceWaitIdle(ctx_.device);
        destroy_targets();
        width_ = width;
        height_ = height;
        vis_ = ctx_.make_buffer(uint64_t(width) * height * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        shadow_mask_ = ctx_.make_buffer(uint64_t((width + 1) / 2) * ((height + 1) / 2) * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
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
            VkDescriptorBufferInfo mask_info{shadow_mask_.handle, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet m{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            m.dstSet = f.set;
            m.dstBinding = 20;
            m.descriptorCount = 1;
            m.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            m.pBufferInfo = &mask_info;
            vkUpdateDescriptorSets(ctx_.device, 1, &m, 0, nullptr);
        }
        // two history images, ping-ponged by the frame slots.
        if (!history_sampler_) {
            VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
            sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            VK_CHECK(vkCreateSampler(ctx_.device, &sci, nullptr, &history_sampler_));
        }
        for (vk::image& h : history_) {
            h = ctx_.make_image(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT);
            ctx_.submit([&](VkCommandBuffer cmd) {
                vk::transition(cmd, h.handle, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            });
        }
        history_valid_ = false;
        for (size_t k = 0; k < frames_in_flight; ++k) {
            VkDescriptorImageInfo read{history_sampler_, history_[1 - k].view, VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo write{VK_NULL_HANDLE, history_[k].view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            w[0].dstSet = slots_[k].set;
            w[0].dstBinding = 21;
            w[0].descriptorCount = 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo = &read;
            w[1].dstSet = slots_[k].set;
            w[1].dstBinding = 22;
            w[1].descriptorCount = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo = &write;
            vkUpdateDescriptorSets(ctx_.device, 2, w, 0, nullptr);
        }
        // ambient occlusion, half resolution.
        ao_ = ctx_.make_image((width + 1) / 2, (height + 1) / 2, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT);
        // its depth chain (ao_depth.comp): half resolution and three halvings.
        VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ic.imageType = VK_IMAGE_TYPE_2D;
        ic.format = VK_FORMAT_R32_SFLOAT;
        ic.extent = {(width + 1) / 2, (height + 1) / 2, 1};
        ic.mipLevels = ao_depth_levels;
        ic.arrayLayers = 1;
        ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling = VK_IMAGE_TILING_OPTIMAL;
        ic.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ao_depth_ = ctx_.make_image(ic, VK_IMAGE_ASPECT_COLOR_BIT);
        for (uint32_t l = 0; l < ao_depth_levels; ++l) {
            VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vc.image = ao_depth_.handle;
            vc.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vc.format = VK_FORMAT_R32_SFLOAT;
            vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, l, 1, 0, 1};
            VK_CHECK(vkCreateImageView(ctx_.device, &vc, nullptr, &ao_depth_views_[l]));
        }
        ctx_.submit([&](VkCommandBuffer cmd) {
            for (VkImage image : {ao_.handle, ao_depth_.handle}) {
                VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                ib.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                ib.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                ib.image = image;
                ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1;
                di.pImageMemoryBarriers = &ib;
                vkCmdPipelineBarrier2(cmd, &di);
            }
        });
        for (frame_slot& f : slots_) {
            VkDescriptorImageInfo info{VK_NULL_HANDLE, ao_.view, VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo levels[ao_depth_levels];
            for (uint32_t l = 0; l < ao_depth_levels; ++l) levels[l] = {VK_NULL_HANDLE, ao_depth_views_[l], VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo sampled{sampler_, ao_depth_.view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[3] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
                                         {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            w[0].dstSet = w[1].dstSet = w[2].dstSet = f.set;
            w[0].dstBinding = 35;
            w[0].descriptorCount = 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[0].pImageInfo = &info;
            w[1].dstBinding = 36;
            w[1].descriptorCount = ao_depth_levels;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo = levels;
            w[2].dstBinding = 37;
            w[2].descriptorCount = 1;
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[2].pImageInfo = &sampled;
            vkUpdateDescriptorSets(ctx_.device, 3, w, 0, nullptr);
        }
        create_hzb();
        prev_valid_ = false;
        ctx_.destroy(readback_);
        readback_ = ctx_.make_buffer(uint64_t(width) * height * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    }

    struct frame_result {
        gpu_stats stats{};
        // pass 1 culling, pass 1 drawing, pyramids and pass 2, shadow pages, shading, then the
        // total: from the frame that last used this slot.
        double ms[6] = {};  // the last is the total
        double vsm_ms[vsm_profile_steps] = {};  // COLOSSUS_PROFILE_VSM: shadow passes (vsm_profile_names)
        streamer::stats_t streaming;
        double streaming_ms = 0;  // render thread time in the streamer
        bool valid = false;
    };

    // records and submits a frame. with a swapchain image the result is blitted into it, else
    // read_pixels() reads it. returns the gpu's report for the slot's last frame.
    frame_result draw(const gpu_frame& frame_in, VkImage target, VkSemaphore wait, VkSemaphore signal, bool readback) {
        frame_slot& f = slots_[slot_];
        slot_ = (slot_ + 1) % frames_in_flight;
        VK_CHECK(vkWaitForFences(ctx_.device, 1, &f.fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(ctx_.device, 1, &f.fence));

        frame_result result;
        if (f.used) {
            std::memcpy(&result.stats, f.stats.mapped, sizeof(gpu_stats));
            uint64_t ts[timestamp_count];
            // the shadow passes' stamps are written only when profiling.
            const uint32_t written = profile_vsm_ ? timestamp_count : 6;
            if (vkGetQueryPoolResults(ctx_.device, f.queries, 0, written, written * sizeof(uint64_t), ts, sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                const double tick = ctx_.properties.limits.timestampPeriod * 1e-6;
                for (int k = 0; k < 5; ++k) result.ms[k] = (ts[k + 1] - ts[k]) * tick;
                result.ms[5] = (ts[5] - ts[0]) * tick;
                if (profile_vsm_)
                    for (uint32_t k = 0; k < vsm_profile_steps; ++k)
                        result.vsm_ms[k] = (ts[6 + k] - (k == 0 ? ts[3] : ts[6 + k - 1])) * tick;
                result.valid = true;
            }
        }
        // streaming: what the gpu asked for and drew from two frames ago, and pages loaded for
        // it into this slot's staging.
        std::vector<std::pair<uint32_t, float>> wanted;
        if (f.used) {
            streamer_.note_used(static_cast<const uint32_t*>(f.used_readback.mapped));
            const uint32_t* r = static_cast<const uint32_t*>(f.request_readback.mapped);
            const uint32_t n = std::min(r[0], max_requests);
            for (uint32_t k = 0; k < n; ++k) {
                float priority;
                std::memcpy(&priority, &r[4 + 2 * k + 1], 4);
                wanted.push_back({r[4 + 2 * k], priority});
            }
        }
        const uint32_t frame_index = ++frame_counter_;
        const auto service_start = std::chrono::steady_clock::now();
        const std::vector<streamer::copy> uploads =
            streamer_.service(frame_index, std::move(wanted), static_cast<uint8_t*>(f.staging.mapped), upload_bytes_,
                              frame_in.lod_threshold);
        result.streaming_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - service_start).count();
        const std::vector<uint32_t>& table = streamer_.table();
        std::memcpy(static_cast<uint8_t*>(f.staging.mapped) + upload_bytes_, table.data(), table.size() * 4);
        result.streaming = streamer_.stats();
        f.used = true;

        gpu_frame fr = frame_in;
        fr.prev_time = history_valid_ ? prev_time_ : fr.time;
        pose_slots(f, fr.time, fr.prev_time);
        if (sc_.moving) fr.flags |= flag_moving;
        fr.vsm_atlas_side = vsm_side_;
        // traced bounce light needs ray queries that read hit vertices (position fetch).
        if (!ctx_.position_fetch || !ao_rt_pipeline_) fr.flags &= ~flag_gi_rt;
        fr.page_count = static_cast<uint32_t>(sc_.pages.size());

        prev_time_ = fr.time;
        // taa: each frame's projection nudged by a sub-pixel offset (halton 2, 3 over eight
        // frames); taa.comp blends with last frame's image reprojected through last frame's
        // camera, unnudged.
        float unjittered[16];
        std::memcpy(unjittered, fr.view_proj, sizeof unjittered);
        const bool taa = (fr.flags & flag_taa) && fr.debug_mode == 0;
        if (taa) {
            auto halton = [](uint32_t i, uint32_t base) {
                float f = 1, r = 0;
                for (; i > 0; i /= base) { f /= base; r += f * (i % base); }
                return r;
            };
            const uint32_t k = frame_index % 8 + 1;
            const float jx = (halton(k, 2) - 0.5f) * 2 / width_, jy = (halton(k, 3) - 0.5f) * 2 / height_;
            fr.jitter[0] = jx;
            fr.jitter[1] = jy;
            // clip.xy += jitter * clip.w; w is the last row.
            for (int c = 0; c < 4; ++c) {
                fr.view_proj[c * 4 + 0] += jx * fr.view_proj[c * 4 + 3];
                fr.view_proj[c * 4 + 1] += jy * fr.view_proj[c * 4 + 3];
            }
            mat4 vp;
            std::memcpy(vp.m, fr.view_proj, sizeof vp.m);
            const mat4 inv = inverse(vp);
            std::memcpy(fr.inv_view_proj, inv.m, sizeof fr.inv_view_proj);
        }
        std::memcpy(fr.prev_view_proj, prev_view_proj_, sizeof prev_view_proj_);
        fr.taa_valid = history_valid_ && taa;
        std::memcpy(prev_view_proj_, unjittered, sizeof prev_view_proj_);
        history_valid_ = true;
        fr.frame_index = frame_index;
        fr.scene_top = scene_top_;
        std::memcpy(fr.shadow_lod_error, sc_.shadow_lod_error, sizeof fr.shadow_lod_error);
        fr.max_requests = max_requests;
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

        // last frame's passes may still read what this clears.
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 0);
        std::vector<VkBufferCopy> regions;
        for (const streamer::copy& c : uploads) regions.push_back({c.staging_offset, c.pool_offset, c.size});
        if (!regions.empty()) vkCmdCopyBuffer(cmd, f.staging.handle, pool_buffer_.handle, uint32_t(regions.size()), regions.data());
        const VkBufferCopy table_copy{upload_bytes_, 0, table.size() * 4};
        vkCmdCopyBuffer(cmd, f.staging.handle, page_table_.handle, 1, &table_copy);
        vkCmdFillBuffer(cmd, requests_.handle, 0, 16, 0);
        vkCmdFillBuffer(cmd, vis_.handle, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, work_.handle, 0, 32, 0);  // both passes' counts, late counts, big counts
        vkCmdFillBuffer(cmd, cell_lists_.handle, 0, 16, 0);
        if (fr.flags & flag_vsm) {
            vkCmdFillBuffer(cmd, vsm_lists_.handle, 0, 4 * vsm_lists_header, 0);
            vkCmdFillBuffer(cmd, vsm_work_.handle, 0, 16, 0);
        }
        vkCmdFillBuffer(cmd, visible_.handle, 0, 16, 0);
        vkCmdFillBuffer(cmd, f.stats.handle, 0, sizeof(gpu_stats), 0);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &f.set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &f.set, 0, nullptr);
        if (tlas_scratch_address_ && (fr.flags & flag_shadows) && !(fr.flags & flag_vsm)) update_shadow_scene(cmd);
        vk::transition(cmd, depth_.handle, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        // pass 1: what last frame's depth does not hide. then a pyramid from it, and pass 2:
        // what pass 1 wrongly hid. then the pyramid again, for next frame.
        draw_pass(cmd, 0, f.queries);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 2);
        build_hzb(cmd);
        draw_pass(cmd, 1, f.queries);
        build_hzb(cmd);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 3);

        vk::transition(cmd, color_.handle, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
        if ((fr.flags & flag_vsm) && (fr.flags & flag_shadows)) update_shadow_pages(cmd, f);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 4);
        if (fr.flags & flag_ao) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ao_depth_pipeline_);
            for (uint32_t l = 0; l < ao_depth_levels; ++l) {
                const gpu_push push{0, l};
                vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_ALL, 0, sizeof push, &push);
                vkCmdDispatch(cmd, (((width_ + 1) / 2 >> l) + 8) / 8, (((height_ + 1) / 2 >> l) + 8) / 8, 1);
                compute_barrier(cmd);
            }
            const gpu_push back{0, 0};
            vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_ALL, 0, sizeof back, &back);
            // the ray query variant only for ray traced shadows: it costs 0.05 ms even unused.
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              ao_rt_pipeline_ && (!(fr.flags & flag_vsm) || (fr.flags & flag_gi_rt)) ? ao_rt_pipeline_ : ao_pipeline_);
            vkCmdDispatch(cmd, ((width_ + 1) / 2 + 7) / 8, ((height_ + 1) / 2 + 7) / 8, 1);
            compute_barrier(cmd);
        }
        if (shadow_ && (frame_in.flags & flag_shadows) && !(frame_in.flags & flag_full_res_shadows) && !(fr.flags & flag_vsm)) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shadow_);
            vkCmdDispatch(cmd, ((width_ + 1) / 2 + 7) / 8, ((height_ + 1) / 2 + 7) / 8, 1);
            vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shade_);
        vkCmdDispatch(cmd, (width_ + 7) / 8, (height_ + 7) / 8, 1);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        // into this slot's history image, from the other (last frame's).
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, taa_);
        vkCmdDispatch(cmd, (width_ + 7) / 8, (height_ + 7) / 8, 1);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 5);
        const VkImage out = history_[size_t(&f - slots_.data())].handle;
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        if (target) {
            vk::transition(cmd, target, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, 0,
                           VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {int32_t(width_), int32_t(height_), 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[1] = {int32_t(width_), int32_t(height_), 1};
            vkCmdBlitImage(cmd, out, VK_IMAGE_LAYOUT_GENERAL, target,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
            vk::transition(cmd, target, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
        }
        // what this frame asked for and drew from, for the streamer.
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT);
        const VkBufferCopy request_copy{0, 0, f.request_readback.size};
        vkCmdCopyBuffer(cmd, requests_.handle, f.request_readback.handle, 1, &request_copy);
        const VkBufferCopy used_copy{0, 0, f.used_readback.size};
        vkCmdCopyBuffer(cmd, page_used_.handle, f.used_readback.handle, 1, &used_copy);
        if (readback) {
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {width_, height_, 1};
            vkCmdCopyImageToBuffer(cmd, out, VK_IMAGE_LAYOUT_GENERAL, readback_.handle, 1, &copy);
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

    // next frame's acquire semaphore, once its last user is done.
    VkSemaphore image_ready_semaphore() {
        VK_CHECK(vkWaitForFences(ctx_.device, 1, &slots_[slot_].fence, VK_TRUE, UINT64_MAX));
        return slots_[slot_].image_ready;
    }

    // last frame drawn with readback, as 8-bit srgb.
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
    static constexpr uint32_t timestamp_count = 6 + vsm_profile_steps;


    struct frame_slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore image_ready = VK_NULL_HANDLE;
        VkQueryPool queries = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        vk::buffer frame, stats;
        vk::buffer pose_now, pose_prev, pose_info;  // skinning: this frame's and last frame's joints, per slot bounds
        vk::buffer staging;           // pages loaded for this frame, then the page table
        vk::buffer request_readback;  // this frame's requests
        vk::buffer used_readback;     // frame each page was last drawn from
        bool used = false;
    };

    // poses every skinned slot at time t (an animation at its phase) into this frame's
    // buffers, with last frame's joints for motion, and per slot: how far its joints carry the
    // model's bounds and lod bounds from its anchor joint (instance_sphere() in common.glsl),
    // and the fastest a point goes.
    void pose_slots(frame_slot& f, float t, float prev_t) {
        if (sc_.pose_slots.empty()) return;
        float* now = static_cast<float*>(f.pose_now.mapped);
        float* info = static_cast<float*>(f.pose_info.mapped);
        std::vector<float> rows;
        const float dt = t - prev_t;
        const bool have_last = last_pose_.size() == 16ull * sc_.pose_joints;
        uint32_t at = 0;
        for (size_t k = 0; k < sc_.pose_slots.size(); ++k) {
            const scene::pose_slot& ps = sc_.pose_slots[k];
            const skeleton& sk = sc_.skeletons[ps.skeleton];
            const float duration = ps.animation < 0 ? 0.0f : sk.animations[size_t(ps.animation)].duration;
            pose_joints(sk, ps.animation, t + ps.phase * duration, rows);
            const gpu_mesh& m = sc_.meshes[ps.mesh];
            const vec3 c = {m.bounds[0], m.bounds[1], m.bounds[2]};
            const float r = m.bounds[3];
            const vec3 lc = {m.lod_bounds[0], m.lod_bounds[1], m.lod_bounds[2]};
            const float lr = m.lod_bounds[3];
            auto apply = [](const float* q, vec3 p) {
                return vec3(q[0] * p.x + q[1] * p.y + q[2] * p.z + q[3], q[4] * p.x + q[5] * p.y + q[6] * p.z + q[7],
                            q[8] * p.x + q[9] * p.y + q[10] * p.z + q[11]);
            };
            const float* anchor = &rows[12 * size_t(m.skin[3])];
            float grow = 0, lod_grow = 0, speed = 0;
            for (size_t j = 0; j < sk.joints.size(); ++j, ++at) {
                const float* jr = &rows[12 * j];
                float* out = now + 16 * size_t(at);
                std::copy_n(jr, 12, out);
                out[12] = spectral_norm(jr, true);
                // relative to the anchor: |M_j c - M_a c| + |A_j - A_a| r.
                float diff[12];
                for (int i = 0; i < 12; ++i) diff[i] = jr[i] - anchor[i];
                const float apart = spectral_norm(diff, false);
                out[13] = apart;
                out[14] = out[15] = 0;
                grow = std::max(grow, length(apply(jr, c) - apply(anchor, c)) + apart * r);
                lod_grow = std::max(lod_grow, length(apply(jr, lc) - apply(anchor, lc)) + apart * lr);
                if (have_last && dt > 0) {
                    // the change since last frame: the centre's move and the linear part's.
                    const float* q = &last_pose_[16 * size_t(at)];
                    float d[12];
                    for (int i = 0; i < 12; ++i) d[i] = jr[i] - q[i];
                    for (int i = 0; i < 3; ++i) d[4 * i + i] += 1;  // spectral_norm(.., true) takes it off
                    const vec3 dc(d[0] * c.x + d[1] * c.y + d[2] * c.z + d[3] - c.x, d[4] * c.x + d[5] * c.y + d[6] * c.z + d[7] - c.y,
                                  d[8] * c.x + d[9] * c.y + d[10] * c.z + d[11] - c.z);
                    speed = std::max(speed, (length(dc) + spectral_norm(d, true) * r) / dt);
                }
            }
            info[4 * k] = grow;
            info[4 * k + 1] = lod_grow;
            info[4 * k + 2] = speed;
            info[4 * k + 3] = 0;
        }
        const size_t bytes = 64ull * sc_.pose_joints;
        // last frame's joints: kept from last frame, or this frame's on the first.
        std::memcpy(f.pose_prev.mapped, have_last ? last_pose_.data() : now, bytes);
        last_pose_.assign(now, now + 16ull * sc_.pose_joints);
    }

    vk::context& ctx_;
    const scene& sc_;
    uint32_t width_ = 0, height_ = 0;
    std::array<frame_slot, frames_in_flight> slots_;
    uint32_t slot_ = 0;

    vk::buffer clusters_, page_table_, pool_buffer_, page_used_, requests_, request_stamp_, shadow_positions_, meshes_, instances_;
    vk::buffer page_skins_;
    std::vector<float> last_pose_;  // last frame's joints (16 floats each), for motion
    vk::buffer work_, visible_, draw_args_, vis_, readback_, late_instances_, late_clusters_, cells_, cell_lists_;
    // virtual shadow maps: see vsm.glsl.
    uint32_t vsm_side_ = 1;
    vk::buffer vsm_entries_, vsm_phys_, vsm_lists_, vsm_atlas_, vsm_work_, vsm_visible_, vsm_args_, moving_instances_;
    VkPipeline vsm_mark_ = VK_NULL_HANDLE, vsm_alloc_ = VK_NULL_HANDLE, vsm_clear_ = VK_NULL_HANDLE, vsm_args_pipeline_ = VK_NULL_HANDLE,
               vsm_instance_ = VK_NULL_HANDLE, vsm_expand_ = VK_NULL_HANDLE, vsm_cluster_ = VK_NULL_HANDLE, vsm_raster_ = VK_NULL_HANDLE;

    void compute_barrier(VkCommandBuffer cmd) {
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }

    // updates this frame's shadow pages (vsm.glsl): mark from the visibility buffer, give
    // missing ones physical pages, render those and invalidated ones.
    void update_shadow_pages(VkCommandBuffer cmd, frame_slot& f) {
        // with COLOSSUS_PROFILE_VSM, a timestamp after each step (vsm_profile_names).
        uint32_t stamp = 6;
        auto mark_time = [&] {
            if (profile_vsm_) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, stamp++);
        };
        auto run = [&](VkPipeline p, uint32_t step, uint32_t groups) {
            const gpu_push push{0, step};
            vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_ALL, 0, sizeof push, &push);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p);
            vkCmdDispatch(cmd, std::min(groups, 65535u), (groups + 65534) / 65535, 1);
            compute_barrier(cmd);
        };
        auto run_indirect = [&](VkPipeline p, VkDeviceSize offset) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p);
            vkCmdDispatchIndirect(cmd, vsm_args_.handle, offset);
            compute_barrier(cmd);
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vsm_mark_);
        vkCmdDispatch(cmd, ((width_ + 1) / 2 + 7) / 8, ((height_ + 1) / 2 + 7) / 8, 1);  // a 2x2 block per invocation
        compute_barrier(cmd);
        mark_time();
        if (sc_.moving) run(vsm_alloc_, 3, uint32_t(sc_.moving));
        run(vsm_alloc_, 0, (vsm_slots + 63) / 64);
        run(vsm_alloc_, 1, (vsm_side_ * vsm_side_ + 63) / 64);
        run(vsm_alloc_, 2, (vsm_slots + 63) / 64);
        run(vsm_args_pipeline_, 0, 1);
        mark_time();
        run_indirect(vsm_clear_, 16);
        mark_time();
        run_indirect(vsm_instance_, 0);
        run(vsm_args_pipeline_, 1, 1);
        run_indirect(vsm_expand_, 32);
        mark_time();
        run(vsm_args_pipeline_, 2, 1);
        run_indirect(vsm_cluster_, 48);
        mark_time();
        run(vsm_args_pipeline_, 3, 1);
        run_indirect(vsm_raster_, 64);
        mark_time();
        // pages rendered, and pages that found no physical page.
        const VkBufferCopy counts[2] = {{12, offsetof(gpu_stats, vsm_rendered), 4}, {24, offsetof(gpu_stats, vsm_overflow), 4}};
        vkCmdCopyBuffer(cmd, vsm_lists_.handle, f.stats.handle, 2, counts);
        const VkBufferCopy work_counts{0, offsetof(gpu_stats, vsm_work), 12};
        vkCmdCopyBuffer(cmd, vsm_work_.handle, f.stats.handle, 1, &work_counts);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        const gpu_push back{0, 0};
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_ALL, 0, sizeof back, &back);
    }
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
               hzb_ = VK_NULL_HANDLE, cluster_cull_ = VK_NULL_HANDLE, sw_raster_ = VK_NULL_HANDLE, shadow_ = VK_NULL_HANDLE;
    vk::buffer shadow_mask_;
    VkPipeline taa_ = VK_NULL_HANDLE, expand_ = VK_NULL_HANDLE, cell_cull_ = VK_NULL_HANDLE;
    std::array<vk::image, frames_in_flight> history_;  // slot k writes history_[k], reads the other
    vk::image ao_;                                     // ao.comp's output, half resolution: indirect light, occlusion
    static constexpr uint32_t ao_depth_levels = 4;
    vk::image ao_depth_;                               // ao_depth.comp's chain
    std::array<VkImageView, ao_depth_levels> ao_depth_views_{};
    VkPipeline ao_pipeline_ = VK_NULL_HANDLE, ao_rt_pipeline_ = VK_NULL_HANDLE, ao_depth_pipeline_ = VK_NULL_HANDLE;
    VkSampler history_sampler_ = VK_NULL_HANDLE;
    float prev_view_proj_[16] = {};
    float prev_time_ = 0;
    bool history_valid_ = false;
    bool cull_only_ = false;
    bool profile_vsm_ = std::getenv("COLOSSUS_PROFILE_VSM") != nullptr;
    uint64_t upload_bytes_;
    streamer streamer_;
    uint32_t frame_counter_ = 0;
    float scene_top_ = 0;

    void destroy_targets() {
        ctx_.destroy(vis_);
        ctx_.destroy(shadow_mask_);
        for (vk::image& h : history_) ctx_.destroy(h);
        ctx_.destroy(ao_);
        for (VkImageView& v : ao_depth_views_) {
            if (v) vkDestroyImageView(ctx_.device, v, nullptr);
            v = VK_NULL_HANDLE;
        }
        ctx_.destroy(ao_depth_);
        ctx_.destroy(depth_);
        ctx_.destroy(color_);
        for (VkImageView v : hzb_views_) vkDestroyImageView(ctx_.device, v, nullptr);
        hzb_views_.clear();
        ctx_.destroy(hzb_image_);
    }

    // depth pyramid: level 0 half the screen (rounded up), each level half the last, down to
    // 1x1. one view of all levels for sampling, one per level for writing.
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

    // args.comp for a pass: step 0 sizes cluster culling, step 1 the draw.
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

    // culls and draws a pass into the visibility buffer.
    void draw_pass(VkCommandBuffer cmd, uint32_t pass, VkQueryPool queries) {
        const gpu_push push{pass, 0};
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_ALL, 0, sizeof push, &push);
        // cells first: all in pass 1, those it hid in pass 2.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cell_cull_);
        if (pass == 0) {
            const uint32_t n = (uint32_t(sc_.cells.size()) + 63) / 64;
            vkCmdDispatch(cmd, std::min(n, 65535u), (n + 65534) / 65535, 1);
        } else {
            vkCmdDispatchIndirect(cmd, draw_args_.handle, offsetof(draw_args_layout, late_cell_args));
        }
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        write_args(cmd, pass, 3);
        // then visible cells' instances (and in pass 2 those pass 1 hid): sized by args.comp.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, instance_cull_);
        vkCmdDispatchIndirect(cmd, draw_args_.handle, offsetof(draw_args_layout, instance_args) + 16 * pass);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        // instances with many work items get a workgroup each.
        write_args(cmd, pass, 2);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, expand_);
        vkCmdDispatchIndirect(cmd, draw_args_.handle, offsetof(draw_args_layout, big_args) + 16 * pass);
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
        clusters_ = ctx_.upload(sc_.clusters, ssbo);
        const VkBufferUsageFlags dst = VK_BUFFER_USAGE_TRANSFER_DST_BIT, src = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        const uint64_t pages = sc_.pages.size();
        // the page table (rewritten each frame), then each page's shared bounds (common.glsl).
        std::vector<uint32_t> table(pages, UINT32_MAX);
        table.resize(pages + sc_.shared_bounds.size() * 5);
        std::memcpy(table.data() + pages, sc_.shared_bounds.data(), sc_.shared_bounds.size() * sizeof(page_bounds));
        page_table_ = ctx_.upload(table, ssbo | dst);
        pool_buffer_ = ctx_.make_buffer(streamer_.pool_bytes(), ssbo | dst, false);
        page_used_ = ctx_.make_buffer(pages * 4, ssbo | src | dst, false);
        request_stamp_ = ctx_.make_buffer(pages * 4, ssbo | dst, false);
        requests_ = ctx_.make_buffer(16 + uint64_t(max_requests) * 8, ssbo | src | dst, false);
        ctx_.submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, page_used_.handle, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, request_stamp_.handle, 0, VK_WHOLE_SIZE, 0);
        });
        // top of the scene: each instance's bounding sphere, placed.
        for (const gpu_instance& inst : sc_.instances) {
            const gpu_mesh& m = sc_.meshes[inst.mesh];
            const float y = inst.rows[1][0] * m.bounds[0] + inst.rows[1][1] * m.bounds[1] + inst.rows[1][2] * m.bounds[2] + inst.rows[1][3];
            scene_top_ = std::max(scene_top_, y + m.bounds[3] * inst.scale);
        }
        if (ctx_.ray_query) {
            shadow_positions_ = ctx_.upload(sc_.shadow_positions, ssbo | as_input);
            build_shadow_scene();
        }
        meshes_ = ctx_.upload(sc_.meshes, ssbo);
        page_skins_ = ctx_.upload(sc_.page_skins, ssbo);
        instances_ = ctx_.upload(sc_.instances, ssbo);
        cells_ = ctx_.upload(sc_.cells, ssbo);
        const VkBufferUsageFlags filled = ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vsm_entries_ = ctx_.make_buffer(16ull * vsm_slots, filled, false);
        vsm_phys_ = ctx_.make_buffer(8ull * vsm_side_ * vsm_side_, filled, false);
        vsm_lists_ = ctx_.make_buffer(4ull * (vsm_lists_header + 5 * vsm_slots), filled | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
        // two layers: still, moving.
        vsm_atlas_ = ctx_.make_buffer(2 * 4ull * vsm_side_ * vsm_side_ * vsm_page * vsm_page, ssbo, false);
        vsm_work_ = ctx_.make_buffer(16 + 8ull * (max_work_items + 2 * vsm_big_capacity), filled | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
        vsm_visible_ = ctx_.make_buffer(8ull * max_visible, ssbo, false);
        vsm_args_ = ctx_.make_buffer(80, ssbo | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false);
        std::vector<uint32_t> moving;
        for (uint32_t i = 0; i < sc_.instances.size(); ++i)
            if (sc_.instances[i].anim) moving.push_back(i);
        if (moving.empty()) moving.push_back(0);  // a buffer cannot be empty; unread then
        moving_instances_ = ctx_.upload(moving, ssbo);
        // all slots empty, all physical pages free (vsm_none is all ones).
        ctx_.submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, vsm_entries_.handle, 0, VK_WHOLE_SIZE, 0xffffffffu);
            vkCmdFillBuffer(cmd, vsm_phys_.handle, 0, VK_WHOLE_SIZE, 0xffffffffu);
        });
        cell_lists_ = ctx_.make_buffer(16 + 3 * 4 * uint64_t(sc_.cells.size()), ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        work_ = ctx_.make_buffer(32 + 2 * uint64_t(max_work_items) * 8, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        visible_ = ctx_.make_buffer(16 + uint64_t(max_visible) * 8, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        draw_args_ = ctx_.make_buffer(sizeof(draw_args_layout), ssbo | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false);
        // hidden instances, then each pass's instances for expand.comp (three words each).
        late_instances_ = ctx_.make_buffer(4 * 7 * sc_.instances.size(), ssbo, false);
        late_clusters_ = ctx_.make_buffer(uint64_t(max_visible) * 8, ssbo, false);
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = VK_LOD_CLAMP_NONE;
        VK_CHECK(vkCreateSampler(ctx_.device, &sci, nullptr, &sampler_));
        for (frame_slot& f : slots_) {
            f.frame = ctx_.make_buffer(sizeof(gpu_frame), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
            const uint64_t joint_bytes = std::max<uint64_t>(64ull * sc_.pose_joints, 64);
            f.pose_now = ctx_.make_buffer(joint_bytes, ssbo, true);
            f.pose_prev = ctx_.make_buffer(joint_bytes, ssbo, true);
            f.pose_info = ctx_.make_buffer(std::max<uint64_t>(16ull * sc_.pose_slots.size(), 16), ssbo, true);
            f.stats = ctx_.make_buffer(sizeof(gpu_stats), ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
            f.staging = ctx_.make_buffer(upload_bytes_ + pages * 4, src, true);
            f.request_readback = ctx_.make_buffer(16 + uint64_t(max_requests) * 8, dst, true);
            f.used_readback = ctx_.make_buffer(pages * 4, dst, true);
        }
    }

    struct accel {
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        vk::buffer storage;
        VkDeviceAddress address = 0;
    };
    std::vector<accel> blas_;
    accel tlas_, tlas_moving_;  // still instances; moving instances
    vk::buffer shadow_indices_, as_instances_, as_moving_;
    uint32_t moving_entries_ = 0;
    VkAccelerationStructureGeometryKHR tlas_geom_{};
    vk::buffer tlas_scratch_;  // for refits: see update_shadow_scene()
    VkDeviceAddress tlas_scratch_address_ = 0;
    VkPipeline tlas_update_ = VK_NULL_HANDLE;

    // moves the shadow scene's moving instances: tlas.comp writes transforms, the top level
    // refits in place. small periodic motion keeps the tree good.
    void update_shadow_scene(VkCommandBuffer cmd) {
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tlas_update_);
        const uint32_t n = (moving_entries_ + 63) / 64;
        vkCmdDispatch(cmd, std::min(n, 65535u), (n + 65534) / 65535, 1);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_SHADER_READ_BIT);
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        info.srcAccelerationStructure = info.dstAccelerationStructure = tlas_moving_.handle;
        info.geometryCount = 1;
        info.pGeometries = &tlas_geom_;
        info.scratchData.deviceAddress = tlas_scratch_address_;
        VkAccelerationStructureBuildRangeInfoKHR range{moving_entries_, 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
        ctx_.build_as(cmd, 1, &info, &ranges);
        vk::barrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }

    // builds an acceleration structure over one geometry and waits.
    accel build_accel(VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR& geom, uint32_t primitives,
                      bool updatable = false) {
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type = type;
        // bottom levels keep their vertices readable for --gi rt's hit normals (position fetch).
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                     (updatable ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR : 0) |
                     (ctx_.position_fetch && type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
                          ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR : 0);
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
        if (updatable) {
            tlas_scratch_ = ctx_.make_buffer(sizes.updateScratchSize + align, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
            tlas_scratch_address_ = (tlas_scratch_.address + align - 1) / align * align;
        }
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

    // shadow ray targets: per model a bottom level over its finest cut within the budget
    // (shadow_cut()), and top levels over the instances. rays skip back faces: sunlight() in
    // shade.comp.
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
            tris.vertexData.deviceAddress = shadow_positions_.address + uint64_t(m.first_vertex) * 12;
            tris.vertexStride = 12;
            tris.maxVertex = m.vertex_count - 1;
            tris.indexType = VK_INDEX_TYPE_UINT32;
            tris.indexData.deviceAddress = shadow_indices_.address + uint64_t(m.first_index) * 4;
            blas_.push_back(build_accel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geom, m.index_count / 3));
            triangles += m.index_count / 3;
        }
        // each instance three times, one per shadow copy, each under its own mask bit: a ray
        // sees only the copy it asks for. still instances in a top level built once, moving ones
        // in another refitted each frame: a refit costs by the whole structure's size.
        std::vector<VkAccelerationStructureInstanceKHR> still, moving;
        for (size_t k = 0; k < sc_.instances.size(); ++k)
            for (uint32_t lod = 0; lod < shadow_lods; ++lod) {
                VkAccelerationStructureInstanceKHR ai{};
                std::memcpy(ai.transform.matrix, sc_.instances[k].rows, sizeof ai.transform.matrix);
                ai.instanceCustomIndex = static_cast<uint32_t>(k);  // tlas.comp reads it back
                ai.mask = 1u << lod;
                ai.flags = 0;
                ai.accelerationStructureReference = blas_[sc_.instances[k].mesh * shadow_lods + lod].address;
                (sc_.instances[k].anim ? moving : still).push_back(ai);
            }
        auto top_level = [&](const vk::buffer& b) {
            VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            geom.geometry.instances.data.deviceAddress = b.address;
            return geom;
        };
        const VkBufferUsageFlags input = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        if (still.empty()) still.push_back({});  // one empty instance (mask 0): nothing to hit
        as_instances_ = ctx_.upload(still, input);
        tlas_ = build_accel(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, top_level(as_instances_), uint32_t(still.size()));
        if (!moving.empty()) {
            as_moving_ = ctx_.upload(moving, input | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            moving_entries_ = uint32_t(moving.size());
            tlas_geom_ = top_level(as_moving_);
            tlas_moving_ = build_accel(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tlas_geom_, moving_entries_, true);
        }
        std::printf("shadows: %s triangles over %zu models, built in %.2f s\n", human(double(triangles)).c_str(),
                    sc_.shadow_meshes.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }

    void create_descriptors() {
        std::vector<VkDescriptorSetLayoutBinding> b;
        for (uint32_t i = 0; i <= 41; ++i) {
            if ((i == 18 || i == 23 || i == 24) && !ctx_.ray_query) continue;
            VkDescriptorSetLayoutBinding x{};
            x.binding = i;
            x.descriptorCount = i == 17 ? max_hzb_levels : i == 36 ? ao_depth_levels : 1;
            x.descriptorType = i == 0                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                             : i == 13 || i == 17 || i == 22 || i == 35 || i == 36 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                             : i == 16 || i == 21 || i == 37 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                             : i == 18 || i == 24    ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                                                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            x.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
            b.push_back(x);
        }
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = static_cast<uint32_t>(b.size());
        lci.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &set_layout_));

        const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32 * frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, (3 + max_hzb_levels + ao_depth_levels) * frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * frames_in_flight},
                                              {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2 * frames_in_flight}};
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
            const vk::buffer* buffers[] = {&f.frame,   &clusters_, &page_table_, &pool_buffer_,   &page_used_,
                                           &requests_, &meshes_,   &instances_,  &work_,          &visible_,
                                           nullptr,    &f.stats,   &draw_args_,  nullptr,         &late_instances_,
                                           &late_clusters_, nullptr, nullptr,   nullptr,         &request_stamp_,
                                           nullptr,         nullptr, nullptr,   as_moving_.handle ? &as_moving_ : nullptr,
                                           nullptr,         &cells_, &cell_lists_, &vsm_entries_, &vsm_phys_, &vsm_lists_,
                                           &vsm_atlas_,     &vsm_work_, &vsm_visible_, &vsm_args_, &moving_instances_,
                                           nullptr,         nullptr,   nullptr,       &f.pose_now, &f.pose_prev,
                                           &f.pose_info,    &page_skins_};
            VkDescriptorBufferInfo infos[42];
            std::vector<VkWriteDescriptorSet> writes;
            for (uint32_t i = 0; i < 42; ++i) {
                if (!buffers[i]) continue;  // visibility buffer and output image: written by resize()
                infos[i] = {buffers[i]->handle, 0, VK_WHOLE_SIZE};
                VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.dstSet = f.set;
                w.dstBinding = i;
                w.descriptorCount = 1;
                w.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w.pBufferInfo = &infos[i];
                writes.push_back(w);
            }
            // the moving top level, or with nothing moving the still one again (skipped:
            // flag_moving).
            VkWriteDescriptorSetAccelerationStructureKHR as_info[2] = {
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &tlas_.handle},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1,
                 tlas_moving_.handle ? &tlas_moving_.handle : &tlas_.handle}};
            for (uint32_t k = 0; k < 2 && ctx_.ray_query; ++k) {
                VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.pNext = &as_info[k];
                w.dstSet = f.set;
                w.dstBinding = k == 0 ? 18 : 24;
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
        expand_ = compute_pipeline(expand_spv, sizeof expand_spv);
        cell_cull_ = compute_pipeline(cell_cull_spv, sizeof cell_cull_spv);
        ao_pipeline_ = compute_pipeline(ao_spv, sizeof ao_spv);
        if (ctx_.ray_query) ao_rt_pipeline_ = compute_pipeline(ao_rt_spv, sizeof ao_rt_spv);
        ao_depth_pipeline_ = compute_pipeline(ao_depth_spv, sizeof ao_depth_spv);
        vsm_mark_ = compute_pipeline(vsm_mark_spv, sizeof vsm_mark_spv);
        vsm_alloc_ = compute_pipeline(vsm_alloc_spv, sizeof vsm_alloc_spv);
        vsm_clear_ = compute_pipeline(vsm_clear_spv, sizeof vsm_clear_spv);
        vsm_args_pipeline_ = compute_pipeline(vsm_args_spv, sizeof vsm_args_spv);
        vsm_instance_ = compute_pipeline(vsm_instance_spv, sizeof vsm_instance_spv);
        vsm_expand_ = compute_pipeline(vsm_expand_spv, sizeof vsm_expand_spv);
        vsm_cluster_ = compute_pipeline(vsm_cluster_spv, sizeof vsm_cluster_spv);
        vsm_raster_ = compute_pipeline(vsm_raster_spv, sizeof vsm_raster_spv);
        if (ctx_.ray_query) tlas_update_ = compute_pipeline(tlas_spv, sizeof tlas_spv);
        args_ = compute_pipeline(args_spv, sizeof args_spv);
        shade_ = ctx_.ray_query ? compute_pipeline(shade_rt_spv, sizeof shade_rt_spv) : compute_pipeline(shade_spv, sizeof shade_spv);
        if (ctx_.ray_query) shadow_ = compute_pipeline(shadow_spv, sizeof shadow_spv);
        taa_ = compute_pipeline(taa_spv, sizeof taa_spv);
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
        // scans have holes, showing the inside: draw both sides.
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;  // reversed depth
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

}  // namespace viewer
