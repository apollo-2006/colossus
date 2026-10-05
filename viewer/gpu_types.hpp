#pragma once
// what the cpu and the shaders share: sizes, limits, the frame, mesh, instance and cell layouts
// (common.glsl declares them too), flags and the debug views.
#include <cstddef>
#include <cstdint>

namespace viewer {

constexpr uint32_t frames_in_flight = 2;
// vsm.glsl's sizes.
constexpr uint32_t vsm_levels = 14, vsm_window = 32, vsm_page = 128, vsm_slots = vsm_levels * vsm_window * vsm_window;
constexpr uint32_t vsm_profile_steps = 6;
constexpr const char* vsm_profile_names[vsm_profile_steps] = {"mark", "alloc", "clear", "instances", "clusters", "raster"};
constexpr uint32_t vsm_lists_header = 8 + 8 * vsm_levels + 2 * vsm_levels * vsm_window, vsm_big_capacity = 65536;
constexpr uint32_t max_work_items = 1u << 22;
constexpr uint32_t max_visible = 1u << 22;  // leaves 7 bits for the triangle in a 32-bit id, and 3 to spare
constexpr const char* mode_names[] = {"shaded", "clusters", "triangles", "LOD level", "groups", "instances", "holes", "rasterizer",
                                      "shadow levels", "ambient occlusion", "coverage"};
constexpr uint32_t mode_count = 11;

// as common.glsl declares them (scalar layout).
struct gpu_mesh {
    uint32_t first_cluster, cluster_count;
    float shadow_error;
    uint32_t pad1;
    float bounds[4];
    float lod_bounds[4];
    float grid[4];  // grid point 0 and step: see paged_file.hpp
    // textured: first material in the material table, material count (0: untextured), bit 0 of
    // the third: some material is double sided (no back face culling)
    uint32_t texture[4];
    float uv_map[4];  // texture coordinates' range: u min, v min, extent (paged_file.hpp)
    // skinned: joints (0: none), first pose slot, its first joint in the pose buffers, anchor joint
    uint32_t skin[4];
};
struct gpu_instance {
    float rows[3][4];
    uint32_t mesh;
    float scale;
    uint32_t material;  // into shade.comp's materials
    uint32_t anim;      // see animate() in common.glsl
};
// cells of up to 64 neighbouring instances, culled first (cell_cull.comp): a sphere, and
// consecutive instances.
struct gpu_cell {
    float center[3], radius;
    uint32_t first, count;
    uint32_t pad[2];
};
constexpr uint32_t cell_side = 8;

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
    uint32_t frame_index;
    float view[16];
    float prev_view[16];
    float p00, p11;
    uint32_t hzb_width, hzb_height, hzb_levels;
    float sw_max_pixels;
    uint32_t max_requests;
    float scene_top;
    float prev_time;
    uint32_t vsm_atlas_side;
    float shadow_lod_error[4];
    float prev_view_proj[16];
    uint32_t taa_valid;
    float jitter[2];
    uint32_t page_count;  // where the page table's shared bounds start
    float ground[4];      // the ground's albedo (rgb) and the fog's density (per unit of distance)
};
struct gpu_stats {
    uint32_t instances_visible, work_items, clusters_tested, clusters_drawn, triangles_drawn;
    uint32_t work_overflow, visible_overflow;
    uint32_t instances_occluded, clusters_occluded, clusters_late, clusters_software;
    uint32_t vsm_rendered, vsm_overflow;  // copied from vsm.glsl's lists
    uint32_t vsm_work, vsm_big, vsm_visible;
};
// common.glsl's DrawArgs: size and argument offsets.
struct draw_args_layout {
    uint32_t cull_args[2][4], draw_args[2][4], sw_args[2][4];
    uint32_t pass_start[3], sw_pass_start[3];
    uint32_t late_cell_args[4];
    uint32_t instance_args[2][4];
    uint32_t big_args[2][4];
};
static_assert(sizeof(draw_args_layout) == 200);

struct gpu_push {
    uint32_t pass, level;
};

constexpr uint32_t flag_cone_culling = 1, flag_frustum_culling = 2, flag_wireframe = 4, flag_occlusion = 8,
                   flag_prev_valid = 16, flag_software_raster = 32, flag_shadows = 64, flag_full_res_shadows = 128, flag_taa = 256,
                   flag_moving = 512,  // some instances move: see animate() in common.glsl
                   flag_vsm = 1024,    // virtual shadow maps (vsm.glsl), not rays
                   flag_ao = 2048,     // ambient occlusion (ao.comp)
                   flag_soft_shadows = 4096,  // contact-hardening penumbras
                   flag_bounce = 8192,        // light bounced off the ground (ambient_light() in shade.comp)
                   flag_gi_rt = 16384,        // bounce light traced in world space (traced_light() in ao.comp)
                   flag_no_grid = 32768;      // the ground without its grid lines (a scene's `grid off`)
// ray traced shadows use the finest cut within each budget; far surfaces use coarser ones
// (trace_surface() in surface.glsl).
constexpr size_t shadow_budgets[] = {1u << 18, 1u << 15, 1u << 12};
constexpr uint32_t shadow_lods = 3;
constexpr uint32_t max_hzb_levels = 16;
constexpr uint32_t max_requests = 1u << 14;  // pages the gpu may ask for per frame

}  // namespace viewer
