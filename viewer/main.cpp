// colossus viewer: draws instanced cluster lod hierarchies (.cgeo from colossus_build) with mesh
// shaders and a compute rasterizer on vulkan.
//
// each frame:
//   1. cell_cull.comp, then instance_cull.comp: drop cells and instances out of view or hidden;
//      binary search the clusters an instance might draw, as work items of 64.
//   2. cluster_cull.comp: lod cut, frustum, normal cone, occlusion. survivors go to a visible
//      list.
//   3. draw.mesh + vis.frag for big clusters, sw_raster.comp for small: depth and triangle into
//      a 64-bit visibility buffer by atomic max.
// steps 1 to 3 run twice: against last frame's depth pyramid, then a fresh one.
//   4. shadow pages (vsm_*.comp), shade.comp, taa.comp.
//
//     colossus --model models/lucy.cgeo --grid 10
//     colossus --model a.cgeo --model b.cgeo --grid 40 --headless --frames 60 --screenshot out.png
#include "camera.hpp"
#include "options.hpp"
#include "png.hpp"
#include "renderer.hpp"
#include "scene.hpp"
#include "swapchain.hpp"
#include "vk.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace viewer {
namespace {

struct view_state {
    camera cam;
    camera cull_cam;  // cam unless frozen
    bool frozen = false;
    float threshold = 1;
    uint32_t mode = 0;
    uint32_t flags = flag_cone_culling | flag_frustum_culling | flag_occlusion | flag_software_raster | flag_shadows | flag_taa | flag_ao | flag_soft_shadows | flag_bounce;
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
    f.p11 = -proj.at(1, 1);  // the projection flips y for vulkan; the sphere test wants it upright
    f.flags = v.flags;
    // depth from the drawing camera says nothing about a frozen culling camera.
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
    if (key == GLFW_KEY_X) v->flags ^= flag_taa;
    if (key == GLFW_KEY_G) v->flags ^= flag_ao;
    if (key == GLFW_KEY_B) v->flags ^= flag_bounce;
    if (key == GLFW_KEY_I) v->flags ^= flag_gi_rt;
    if (key == GLFW_KEY_J) v->flags ^= flag_soft_shadows;
    if (key == GLFW_KEY_0) v->mode = 9;
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
    std::snprintf(b, sizeof b, "%.2f ms (cull %.2f, raster %.2f, pass 2 %.2f, shadow pages %.2f, shade %.2f) | %s tris, %s clusters (%s software, %s late) | %u/%zu instances | %.3gpx | %s%s%s%s%s%s",
                  r.ms[5], r.ms[0], r.ms[1], r.ms[2], r.ms[3], r.ms[4], human(r.stats.triangles_drawn).c_str(),
                  human(r.stats.clusters_drawn).c_str(), human(r.stats.clusters_software).c_str(), human(r.stats.clusters_late).c_str(), r.stats.instances_visible,
                  instances, v.threshold,
                  mode_names[v.mode], v.frozen ? " | FROZEN" : "", (v.flags & flag_cone_culling) ? "" : " | no cone",
                  (v.flags & flag_occlusion) ? "" : " | no occlusion", (v.flags & flag_software_raster) ? "" : " | no software raster",
                  (r.stats.work_overflow || r.stats.visible_overflow) ? " | OVERFLOW" : "");
    std::string line = b;
    if (v.flags & flag_vsm) {
        char sp[160];
        std::snprintf(sp, sizeof sp, " | shadow pages: %u rendered (%u clusters), %u short", r.stats.vsm_rendered, r.stats.vsm_visible,
                      r.stats.vsm_overflow);
        line += sp;
    }
    const auto& st = r.streaming;
    char m[160];
    std::snprintf(m, sizeof m, " | pages %u/%u%s", st.resident, st.slots, st.waiting ? " (streaming)" : "");
    return line + m;
}


}  // namespace
}  // namespace viewer

using namespace viewer;

int main(int argc, char** argv) {
    try {
        const options opt = parse(argc, argv);
        scene sc;
        make_scene(opt, sc);
        std::printf("%zu instances (%zu moving), %s triangles at full detail\n", sc.instances.size(), sc.moving,
                    human(double(sc.instanced_triangles)).c_str());

        view_state v;
        v.threshold = opt.threshold;
        v.mode = opt.mode;
        if (opt.wireframe) v.flags |= flag_wireframe;
        if (opt.full_res_shadows) v.flags |= flag_full_res_shadows;
        v.flags &= ~opt.disable;
        if (opt.vsm) v.flags |= flag_vsm;
        if (opt.gi_rt) v.flags |= flag_gi_rt;
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
            renderer r(ctx, sc, opt.width, opt.height, opt.pool_mb << 20, opt.upload_mb << 20, opt.loader_threads, opt.prefetch, opt.cull_only,
                       opt.vsm ? opt.vsm_side : 1, opt.merge_reads);
            std::vector<renderer::frame_result> results;
            const int total = opt.warmup + opt.frames;
            int settled = 0;  // frame after the last that asked for or waited on a page
            uint64_t streamed = 0;
            uint32_t evicted = 0, most_resident = 0, reads = 0;
            std::vector<double> service_ms;
            const auto loop_start = std::chrono::steady_clock::now();
            double settled_ms = 0;
            for (int i = 0; i < total + 2; ++i) {
                if (opt.fly != 0 && i > 0) {
                    v.cam.eye += v.cam.forward() * opt.fly;
                    v.cam.yaw += 0.004f;
                    v.cull_cam = v.cam;
                }
                const bool last = i == total - 1;
                // time at 60 frames a second, so motion repeats run to run.
                auto res = r.draw(frame_data(v, opt.width, opt.height, i / 60.0f), VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                  (last && !opt.screenshot.empty()) || (!opt.record.empty() && i >= opt.warmup && i < total));
                streamed += res.streaming.bytes_loaded;
                evicted += res.streaming.evicted;
                reads += res.streaming.reads;
                service_ms.push_back(res.streaming_ms);
                most_resident = std::max(most_resident, res.streaming.resident);
                if (std::getenv("COLOSSUS_TRACE_STREAMING"))
                    std::printf("frame %d at %.0f ms: requests %u, loaded %u (%.1f MB), waiting %u, resident %u, drawn %u clusters, shadow pages %u (%u clusters)\n", i,
                                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_start).count(),
                                res.streaming.requested, res.streaming.loaded, res.streaming.bytes_loaded / 1048576.0,
                                res.streaming.waiting, res.streaming.resident, res.stats.clusters_drawn, res.stats.vsm_rendered,
                                res.stats.vsm_visible);
                // settled: nothing asked for or waiting from here on (the first frames ask
                // nothing only because no requests are back yet).
                if (res.streaming.requested != 0 || res.streaming.waiting != 0) {
                    settled = i + 1;
                    settled_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_start).count();
                }
                if (res.valid && i >= opt.warmup + 2) results.push_back(res);
                if (!opt.record.empty() && i >= opt.warmup && i < total) {
                    char name[32];
                    std::snprintf(name, sizeof name, "/frame_%05d.png", i - opt.warmup);
                    png::write_rgb(opt.record + name, opt.width, opt.height, r.read_pixels());
                }
                if (last && !opt.screenshot.empty()) {
                    png::write_rgb(opt.screenshot, opt.width, opt.height, r.read_pixels());
                    std::printf("wrote %s\n", opt.screenshot.c_str());
                }
            }
            vkDeviceWaitIdle(ctx.device);
            std::printf("streamed %.0f MB of %.0f MB on disk in %u reads, evicted %u pages, at most %u resident; %s\n",
                        streamed / 1048576.0, sc.total_page_bytes / 1048576.0, reads, evicted, most_resident,
                        settled < total ? ("settled after " + std::to_string(settled) + " frames, " + std::to_string(std::lround(settled_ms)) + " ms").c_str()
                                        : "still streaming");
            std::sort(service_ms.begin(), service_ms.end());
            std::printf("render thread in the streamer: median %.3f ms, 99th percentile %.3f ms, worst %.3f ms (%s)\n",
                        service_ms[service_ms.size() / 2], service_ms[service_ms.size() * 99 / 100], service_ms.back(),
                        opt.loader_threads ? (std::to_string(opt.loader_threads) + " loader threads").c_str() : "loads on the render thread");
            if (!results.empty()) {
                // median frame by total time.
                std::sort(results.begin(), results.end(), [](auto& a, auto& b) { return a.ms[5] < b.ms[5]; });
                std::printf("%s\n", stats_line(results[results.size() / 2], v, sc.instances.size()).c_str());
                if (std::getenv("COLOSSUS_PROFILE_VSM")) {
                    std::printf("shadow passes, medians:");
                    for (uint32_t k = 0; k < vsm_profile_steps; ++k) {
                        std::vector<double> m;
                        for (const auto& r : results) m.push_back(r.vsm_ms[k]);
                        std::sort(m.begin(), m.end());
                        std::printf(" %s %.3f", vsm_profile_names[k], m[m.size() / 2]);
                    }
                    std::printf(" ms\n");
                }
                const auto& s = results[results.size() / 2].stats;
                std::printf("clusters tested %s, work items %s, hidden by last frame %s, instances hidden %u\n",
                            human(s.clusters_tested).c_str(), human(s.work_items).c_str(), human(s.clusters_occluded).c_str(),
                            s.instances_occluded);
            }
            return vk::validation_errors ? 1 : 0;
        }

        if (!glfwInit()) throw std::runtime_error("glfwInit failed");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(opt.width, opt.height, "colossus", nullptr, nullptr);
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
            renderer r(ctx, sc, swap.extent.width, swap.extent.height, opt.pool_mb << 20, opt.upload_mb << 20, opt.loader_threads, opt.prefetch,
                       false, opt.vsm ? opt.vsm_side : 1, opt.merge_reads);
            std::printf("keys: wasd/qe move, drag to look, scroll for speed, shift to hurry\n"
                        "      1-9, 0 view (shaded, clusters, triangles, lod level, groups, instances, holes, rasterizer,\n"
                        "      shadow levels, ambient occlusion)\n"
                        "      [ ] lod threshold, f freeze culling, c cone, v frustum, o occlusion culling,\n"
                        "      r software rasterizer, h shadows, j soft shadows, x antialiasing, g ambient occlusion,\n"
                        "      t wireframe, p print camera\n");

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
                    glfwSetWindowTitle(window, ("colossus | " + stats_line(res, v, sc.instances.size())).c_str());
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
