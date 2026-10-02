// shared by every viewer shader: resources as main.cpp writes them, culling and
// lod tests.
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// gpu_cluster (include/geometry_file.hpp) as paged: `group` is its page,
// `creator` the page of the finer clusters it stands for, offsets are words
// into its page. load_cluster() unpacks it.
struct Cluster {
    vec3 center;  // culling bounds
    float radius;
    vec3 cone_axis;
    float cone_cutoff;
    vec3 lod_center;
    float lod_radius;
    vec3 parent_center;
    float parent_radius;
    float lod_error;
    float parent_error;
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
    uint level;
    uint group;
    uint creator;
    uvec3 origin;  // corner on the model's grid
};

// as stored: packed_cluster (include/paged_file.hpp).
struct PackedCluster {
    vec3 center;
    float radius;
    uint cone;     // axis u (11) | v (11) << 11 | cutoff (10) << 22
    uint offsets;  // vertex_offset | triangle_offset << 16
    uint level;    // level word | vertex_count << 24
    uint group;
    uint creator;
    uint origin[3];  // 24 bits each, origin[0] | triangle_count << 24
};

// a loaded model: clusters[first_cluster, + cluster_count).
struct Mesh {
    uint first_cluster;
    uint cluster_count;
    float shadow_error;  // how far the ray traced shadow copy strays
    uint pad1;
    vec4 bounds;      // xyz centre, w radius
    vec4 lod_bounds;  // holds every lod sphere
    vec4 grid;        // position snapping: xyz grid point 0, w step
};

// a model placed by rotation, uniform scale, translation.
struct Instance {
    vec4 rows[3];  // 3x4 to-world, by rows
    uint mesh;
    float scale;
    uint material;  // into shade.comp's materials
    uint anim;      // 0: still. else moving (animate()): phase in low 8 bits, bit 8 reverses
};

layout(set = 0, binding = 0, scalar) uniform Frame {
    mat4 view_proj;
    // culling and lod use their own camera, which can freeze to watch from
    // outside.
    vec4 cull_planes[5];  // world space: inside where dot(xyz, p) + w >= 0
    vec4 cull_origin;
    vec4 origin;          // drawing camera
    mat4 inv_view_proj;
    uint width, height;
    uint instance_count;
    uint flags;           // flag_* below
    float lod_scale;      // pixels per unit at distance 1: height / (2 tan(fov / 2))
    float lod_threshold;  // allowed error on screen, pixels
    float near_z;
    uint debug_mode;
    uint max_work_items;
    uint max_visible;
    float time;           // seconds, for motion
    uint frame_index;  // from 1: stamps pages used and requested
    // occlusion: the depth pyramid (hzb) and its cameras. pass 1 tests last
    // frame's pyramid from last frame's camera; pass 2 this frame's, built
    // after pass 1.
    mat4 view;
    mat4 prev_view;
    float p00, p11;  // projection scale, x and y
    uint hzb_width, hzb_height, hzb_levels;
    float sw_max_pixels;  // clusters smaller on screen go to the software rasterizer
    uint max_requests;
    float scene_top;  // highest point of any instance: shadow rays stop above it
    float prev_time;      // last frame's time
    uint vsm_atlas_side;  // physical shadow pages a side (vsm.glsl)
    vec4 shadow_lod_error;  // each shadow copy's largest error over the models
    mat4 prev_view_proj;    // last frame's, unjittered: for taa.comp
    uint taa_valid;         // last frame's image fits this one
    vec2 jitter;            // this frame's sub-pixel offset, clip space
    uint page_count;  // page_table's entries; its shared bounds follow
} frame;

const uint flag_cone_culling = 1u;
const uint flag_frustum_culling = 2u;
const uint flag_wireframe = 4u;
const uint flag_occlusion = 8u;
const uint flag_prev_valid = 16u;  // last frame's pyramid fits this frame
const uint flag_software_raster = 32u;
const uint flag_shadows = 64u;
const uint flag_full_res_shadows = 128u;  // every pixel traces its own ray
const uint flag_taa = 256u;
const uint flag_ao = 2048u;  // ambient occlusion (ao.comp)
const uint flag_soft_shadows = 4096u;  // contact-hardening penumbras (vsm_lookup())
const uint flag_vsm = 1024u;  // virtual shadow maps (vsm.glsl), not rays
const uint flag_moving = 512u;  // some instances move: rays also test shadow_moving

// pass (0 or 1), and the pyramid builder's level.
layout(push_constant, scalar) uniform Push {
    uint pass;
    uint level;
} push;

layout(set = 0, binding = 1, scalar) readonly buffer Clusters { PackedCluster packed_clusters[]; };
// streaming (viewer/streamer.hpp): each page's place in the pool (first word,
// or NO_PAGE), the pool, the frame each page was last drawn from, and this
// frame's requests with priority.
const uint NO_PAGE = 0xffffffffu;
// after the frame.page_count entries, each page's shared bounds: centre, radius, error
// (include/paged_file.hpp's page_bounds).
layout(set = 0, binding = 2, scalar) readonly buffer PageTable { uint page_table[]; };

vec3 decode_octahedral(uint uv) {
    const vec2 e = vec2(float(uv & 2047u), float((uv >> 11) & 2047u)) / 2047.0 * 2.0 - 1.0;
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

// page p's shared bounds: xyz centre, w radius; error through page_error().
vec4 page_sphere(uint p) {
    const uint at = frame.page_count + 5u * p;
    return uintBitsToFloat(uvec4(page_table[at], page_table[at + 1u], page_table[at + 2u], page_table[at + 3u]));
}
float page_error(uint p) { return uintBitsToFloat(page_table[frame.page_count + 5u * p + 4u]); }

// for the binary searches over a model's clusters (sorted by it).
float cluster_parent_error(uint i) { return page_error(packed_clusters[i].group); }

Cluster load_cluster(uint i) {
    const PackedCluster p = packed_clusters[i];
    Cluster c;
    c.center = p.center;
    c.radius = p.radius;
    const uint steps = p.cone >> 22;
    c.cone_axis = steps == 1023u ? vec3(0.0, 0.0, 1.0) : decode_octahedral(p.cone);
    c.cone_cutoff = float(steps) / 1023.0 * 2.0 - 1.0;
    c.vertex_offset = p.offsets & 0xffffu;
    c.triangle_offset = p.offsets >> 16;
    c.level = p.level & 0xffffffu;
    c.vertex_count = p.level >> 24;
    c.group = p.group;
    c.creator = p.creator;
    c.origin = uvec3(p.origin[0] & 0xffffffu, p.origin[1], p.origin[2]);
    c.triangle_count = p.origin[0] >> 24;
    const vec4 parent = page_sphere(p.group);
    c.parent_center = parent.xyz;
    c.parent_radius = parent.w;
    c.parent_error = page_error(p.group);
    if (p.creator == NO_PAGE) {
        c.lod_center = p.center;
        c.lod_radius = p.radius;
        c.lod_error = 0.0;
    } else {
        const vec4 lod = page_sphere(p.creator);
        c.lod_center = lod.xyz;
        c.lod_radius = lod.w;
        c.lod_error = page_error(p.creator);
    }
    return c;
}
layout(set = 0, binding = 3, scalar) readonly buffer Pool { uint pool[]; };
layout(set = 0, binding = 4, scalar) buffer PageUsed { uint page_used[]; };
layout(set = 0, binding = 5, scalar) buffer Requests {
    uint request_count;
    uint pad7, pad8, pad9;
    uvec2 requests[];  // (page, priority as float bits)
};
layout(set = 0, binding = 19, scalar) buffer RequestStamp { uint request_stamp[]; };

// a cluster's vertices and triangles from its page at `base`, bit-packed (include/paged_file.hpp):
// a vertex is its offsets from the cluster's corner in bx, by, bz bits and an 11 + 11 bit
// octahedral normal; a triangle three ib-bit indices. the widths ride in the level word.
uint cluster_level(Cluster c) { return c.level & 255u; }

uvec4 cluster_widths(Cluster c) {  // bx, by, bz, ib
    return (uvec4(c.level) >> uvec4(8, 12, 16, 20)) & 15u;
}

// `count` bits (at most 32) at a bit offset into the run starting at word `at`.
uint read_bits(uint at, uint bit, uint count) {
    const uint w = at + (bit >> 5), s = bit & 31u;
    uint v = pool[w] >> s;
    if (s + count > 32u) v |= pool[w + 1u] << (32u - s);
    return count >= 32u ? v : v & ((1u << count) - 1u);
}

vec3 cluster_position(uint base, Cluster c, vec4 grid, uint k) {
    const uvec4 b = cluster_widths(c);
    const uint at = base + c.vertex_offset, bit = k * (b.x + b.y + b.z + 22u);
    const uvec3 d = uvec3(read_bits(at, bit, b.x), read_bits(at, bit + b.x, b.y), read_bits(at, bit + b.x + b.y, b.z));
    return grid.xyz + grid.w * vec3(c.origin + d);
}

vec3 cluster_normal(uint base, Cluster c, uint k) {
    const uvec4 b = cluster_widths(c);
    const uint xyz = b.x + b.y + b.z;
    return decode_octahedral(read_bits(base + c.vertex_offset, k * (xyz + 22u) + xyz, 22u));
}

// triangle t as a | b << 8 | c << 16.
uint cluster_triangle(uint base, Cluster c, uint t) {
    const uint ib = cluster_widths(c).w, mask = (1u << ib) - 1u;
    const uint v = read_bits(base + c.triangle_offset, t * 3u * ib, 3u * ib);
    return (v & mask) | ((v >> ib) & mask) << 8 | ((v >> (2u * ib)) & mask) << 16;
}
layout(set = 0, binding = 6, scalar) readonly buffer Meshes { Mesh meshes[]; };
layout(set = 0, binding = 7, scalar) readonly buffer Instances { Instance instances[]; };

// a moving instance turns on the spot and drifts round a small circle, from the
// time: no uploads. its last frame transform (load_prev_instance) is where last
// frame's pyramid and image have it.
Instance animate(Instance inst, float t) {
    if (inst.anim == 0u) return inst;
    const float phase = float(inst.anim & 255u) * (6.2831853 / 256.0);
    const float turn = phase + t * ((inst.anim & 256u) != 0u ? -0.7 : 0.7);
    const float c = cos(turn), s = sin(turn);
    const vec2 drift = vec2(cos(phase + t * 0.9), sin(phase + t * 0.9)) * 0.2;
    // turn about its own vertical axis, after placement: rows 0 and 2 of the
    // linear part mix.
    const vec3 r0 = inst.rows[0].xyz, r2 = inst.rows[2].xyz;
    inst.rows[0] = vec4(c * r0 + s * r2, inst.rows[0].w + drift.x);
    inst.rows[2] = vec4(-s * r0 + c * r2, inst.rows[2].w + drift.y);
    return inst;
}

// cells of up to 64 neighbouring instances, culled first (cell_cull.comp): a
// sphere over all, consecutive instances.
struct Cell {
    vec3 center;
    float radius;
    uint first;
    uint count;
    uint pad0, pad1;
};
layout(set = 0, binding = 25, scalar) readonly buffer Cells { Cell cells[]; };
// cells pass 1 hid, for pass 2 (cell_list[0, late_cell_count)), then each
// pass's visible cells from cells.length() * (1 + pass).
layout(set = 0, binding = 26, scalar) buffer CellLists {
    uint visible_cell_count[2];
    uint late_cell_count;
    uint cell_pad;
    uint cell_list[];
};

Instance load_instance(uint i) {
    return animate(instances[i], frame.time);
}

Instance load_prev_instance(uint i) {
    return animate(instances[i], frame.prev_time);
}

// 64 clusters of one instance from first. pass 1's in work[0, max_work_items),
// pass 2's next.
struct WorkItem {
    uint instance;
    uint first;  // cluster number
};
layout(set = 0, binding = 8, scalar) buffer WorkItems {
    uint work_count[2];       // per pass
    uint late_instance_count;
    uint late_cluster_count;
    uint big_count[2];        // per pass: instances for expand.comp
    uint pad5, pad6;
    WorkItem work[];
};

// what pass 1 hid, for pass 2: whole instances, and (instance, cluster). after
// instance_count entries, each pass's instances with many work items for
// expand.comp: instance, first cluster, count, three words (big_instance()).
layout(set = 0, binding = 14, scalar) buffer LateInstances { uint late_instances[]; };

uint big_instance(uint pass, uint k) {
    return frame.instance_count * (1u + 3u * pass) + 3u * k;
}
layout(set = 0, binding = 15, scalar) buffer LateClusters { uvec2 late_clusters[]; };

// depth pyramid: level 0 half the screen, each texel the farthest (smallest,
// reversed) depth it covers.
layout(set = 0, binding = 16) uniform sampler2D hzb;

// every cluster drawn this frame, numbered: the visibility buffer stores the
// number and a triangle. hardware clusters fill from the front, software from
// the back.
layout(set = 0, binding = 9, scalar) buffer Visible {
    uint visible_count;
    uint sw_count;
    uint pad3, pad4;
    uvec2 visible[];  // (instance, cluster)
};

uint sw_slot(uint k) {
    return frame.max_visible - 1u - k;
}

// per pixel: reversed depth bits above visible number * 128 + triangle. atomic
// max keeps the nearest; 0 is nothing.
layout(set = 0, binding = 10, scalar) buffer VisBuffer { uint64_t vis[]; };

layout(set = 0, binding = 11, scalar) buffer Stats {
    uint instances_visible;
    uint work_items;
    uint clusters_tested;
    uint clusters_drawn;
    uint triangles_drawn;
    uint work_overflow;
    uint visible_overflow;
    uint instances_occluded;
    uint clusters_occluded;   // in pass 1; some come back in pass 2
    uint clusters_late;       // drawn in pass 2
    uint clusters_software;   // drawn by the software rasterizer
} stats;

// indirect arguments per pass, from args.comp: cluster culling (x, y, z, work
// items), mesh shader draw (x, y, z, first visible), software raster (x, y, z,
// first cluster).
layout(set = 0, binding = 12, scalar) buffer DrawArgs {
    uvec4 cull_args[2];
    uvec4 draw_args[2];
    uvec4 sw_args[2];
    uint pass_start[3];     // visible[pass_start[p], pass_start[p + 1]) is pass p's
    uint sw_pass_start[3];  // same, from the back
    uvec4 late_cell_args;      // pass 2 cell culling: a workgroup per 64 hidden cells
    uvec4 instance_args[2];    // instance culling: a workgroup per visible cell, and in pass 2 per 64 hidden instances
    uvec4 big_args[2];         // expand.comp per pass: a workgroup per big instance
};

vec3 to_world(Instance inst, vec3 p) {
    return vec3(dot(inst.rows[0].xyz, p) + inst.rows[0].w, dot(inst.rows[1].xyz, p) + inst.rows[1].w,
                dot(inst.rows[2].xyz, p) + inst.rows[2].w);
}

// inverse of to_world: rotation and uniform scale, so transpose over scale
// squared.
vec3 from_world(Instance inst, vec3 p) {
    const vec3 d = p - vec3(inst.rows[0].w, inst.rows[1].w, inst.rows[2].w);
    const vec3 c0 = vec3(inst.rows[0].x, inst.rows[1].x, inst.rows[2].x);
    const vec3 c1 = vec3(inst.rows[0].y, inst.rows[1].y, inst.rows[2].y);
    const vec3 c2 = vec3(inst.rows[0].z, inst.rows[1].z, inst.rows[2].z);
    return vec3(dot(c0, d), dot(c1, d), dot(c2, d)) / (inst.scale * inst.scale);
}

vec3 to_world_dir(Instance inst, vec3 v) {
    return vec3(dot(inst.rows[0].xyz, v), dot(inst.rows[1].xyz, v), dot(inst.rows[2].xyz, v));
}

bool sphere_in_frustum(vec3 c, float r) {
    bool inside = true;
    for (int k = 0; k < 5; ++k) inside = inside && dot(frame.cull_planes[k].xyz, c) + frame.cull_planes[k].w >= -r;
    return inside;
}

// pixels an error of world size `error` covers at a sphere's nearest point. a
// group computes it from identical numbers, so it agrees.
float projected_error(vec3 center, float radius, float error) {
    const float d = length(center - frame.cull_origin.xyz) - radius;
    return error * frame.lod_scale / max(d, frame.near_z);
}

// screen rectangle of a view-space sphere (z forward) in [0, 1] texture
// coordinates (mara and mcguire 2013, as in zeux's niagara). false if it
// reaches the near plane.
bool project_sphere(vec3 c, float r, out vec4 aabb) {
    if (c.z < r + frame.near_z) return false;
    const vec3 cr = c * r;
    const float czr2 = c.z * c.z - r * r;
    const float vx = sqrt(c.x * c.x + czr2);
    const float minx = (vx * c.x - cr.z) / (vx * c.z + cr.x);
    const float maxx = (vx * c.x + cr.z) / (vx * c.z - cr.x);
    const float vy = sqrt(c.y * c.y + czr2);
    const float miny = (vy * c.y - cr.z) / (vy * c.z + cr.y);
    const float maxy = (vy * c.y + cr.z) / (vy * c.z - cr.y);
    aabb = vec4(minx * frame.p00, miny * frame.p11, maxx * frame.p00, maxy * frame.p11);
    aabb = aabb.xwzy * vec4(0.5, -0.5, 0.5, -0.5) + vec4(0.5);
    return true;
}

// whether a world-space sphere is surely hidden: nearest point farther than the
// farthest depth in its rectangle. pass 1 from last frame's camera and pyramid,
// pass 2 from this frame's.
bool occluded(vec3 center, float radius) {
    if ((frame.flags & flag_occlusion) == 0u) return false;
    if (push.pass == 0u && (frame.flags & flag_prev_valid) == 0u) return false;
    const mat4 view = push.pass == 0u ? frame.prev_view : frame.view;
    vec3 c = (view * vec4(center, 1.0)).xyz;
    c.z = -c.z;
    vec4 aabb;
    if (!project_sphere(c, radius, aabb)) return false;
    // a pixel's margin for the taa jitter, which the projection leaves out.
    const vec2 screen = vec2(frame.width, frame.height);
    const vec2 lo = max(clamp(aabb.xy, 0.0, 1.0) * screen - 1.0, vec2(0.0));
    const vec2 hi = min(clamp(aabb.zw, 0.0, 1.0) * screen + 0.5, screen - 0.5);
    // finest level where the rectangle spans at most 2x2 texels.
    int level = 0;
    ivec2 t0, t1;
    for (;; ++level) {
        const float s = exp2(float(level + 1));
        t0 = ivec2(floor(lo / s));
        t1 = ivec2(floor(max(hi, lo) / s));
        if (all(lessThanEqual(t1 - t0, ivec2(1))) || level + 1 >= int(frame.hzb_levels)) break;
    }
    const ivec2 size = textureSize(hzb, level);
    t0 = min(t0, size - 1);
    t1 = min(t1, size - 1);
    float far_depth = texelFetch(hzb, t0, level).x;
    far_depth = min(far_depth, texelFetch(hzb, ivec2(t1.x, t0.y), level).x);
    far_depth = min(far_depth, texelFetch(hzb, ivec2(t0.x, t1.y), level).x);
    far_depth = min(far_depth, texelFetch(hzb, t1, level).x);
    const float sphere_depth = frame.near_z / (c.z - radius);
    return sphere_depth < far_depth;
}

// lod test with streaming: draw when own error is within the threshold and the
// parent's is not, or when the finer clusters it stands for are not resident
// (it is the finest copy). it must be resident. finer pages are resident only
// while their stand-ins are (viewer/streamer.hpp), so one level per path.
// wants_finer: drawn only for want of finer clusters.
bool lod_test(Instance inst, Cluster c, out bool wants_finer, out float self_error) {
    const float s = inst.scale;
    self_error = projected_error(to_world(inst, c.lod_center), c.lod_radius * s, c.lod_error * s);
    const bool coarse_enough = projected_error(to_world(inst, c.parent_center), c.parent_radius * s, c.parent_error * s) >
                               frame.lod_threshold;
    const bool finer_resident = c.creator != NO_PAGE && page_table[c.creator] != NO_PAGE;
    wants_finer = self_error > frame.lod_threshold && c.creator != NO_PAGE && !finer_resident;
    return page_table[c.group] != NO_PAGE && coarse_enough && (self_error <= frame.lod_threshold || !finer_resident);
}

// asks for a page, once a frame.
void request_page(uint page, float priority) {
    if (atomicExchange(request_stamp[page], frame.frame_index) == frame.frame_index) return;
    const uint k = atomicAdd(request_count, 1u);
    if (k < frame.max_requests) requests[k] = uvec2(page, floatBitsToUint(priority));
}

// asks for the finer clusters' page.
void request_finer(Cluster c, float priority) { request_page(c.creator, priority); }

const vec3 sun_dir = normalize(vec3(0.75, 0.5, 0.3));  // toward the sun
const vec3 sun_color = vec3(1.0, 0.92, 0.82) * 1.7;

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
