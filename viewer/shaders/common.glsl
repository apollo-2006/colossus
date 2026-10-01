// Shared by every shader of the viewer: its resources, laid out as
// viewer/main.cpp writes them, and the culling and LOD tests.
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// include/geometry_file.hpp's gpu_cluster.
struct Cluster {
    vec3 center;  // Culling bounds
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
};

// One loaded model: its clusters are clusters[first_cluster, + cluster_count).
struct Mesh {
    uint first_cluster;
    uint cluster_count;
    float shadow_error;  // How far the coarser copy that shadows are traced against strays
    uint pad1;
    vec4 bounds;      // xyz center, w radius
    vec4 lod_bounds;  // Contains every LOD sphere of the model
};

// An instance: a model placed by a rotation, uniform scale and translation.
struct Instance {
    vec4 rows[3];  // The 3x4 to-world transform, by rows
    uint mesh;
    float scale;
    uint pad0, pad1;
};

layout(set = 0, binding = 0, scalar) uniform Frame {
    mat4 view_proj;
    // Culling and LOD use their own camera, which can be frozen to watch
    // them from outside.
    vec4 cull_planes[5];  // World space: inside where dot(xyz, p) + w >= 0
    vec4 cull_origin;
    vec4 origin;          // The drawing camera
    mat4 inv_view_proj;
    uint width, height;
    uint instance_count;
    uint flags;           // flag_* below
    float lod_scale;      // Pixels per unit of size at distance 1: height / (2 tan(fov / 2))
    float lod_threshold;  // Allowed error on screen, in pixels
    float near_z;
    uint debug_mode;
    uint max_work_items;
    uint max_visible;
    float time;
    uint pad;
    // Occlusion culling: the depth pyramid (hzb) and the cameras it is
    // tested from. Pass 1 tests against last frame's pyramid, from last
    // frame's camera; pass 2 against this frame's, built after pass 1.
    mat4 view;
    mat4 prev_view;
    float p00, p11;  // Projection scale in x and y
    uint hzb_width, hzb_height, hzb_levels;
    float sw_max_pixels;  // Clusters smaller than this on screen go to the software rasterizer
} frame;

const uint flag_cone_culling = 1u;
const uint flag_frustum_culling = 2u;
const uint flag_wireframe = 4u;
const uint flag_occlusion = 8u;
const uint flag_prev_valid = 16u;  // Last frame's pyramid fits this frame
const uint flag_software_raster = 32u;
const uint flag_shadows = 64u;

// Which pass this is (0 or 1), and for the pyramid builder, which level.
layout(push_constant, scalar) uniform Push {
    uint pass;
    uint level;
} push;

layout(set = 0, binding = 1, scalar) readonly buffer Clusters { Cluster clusters[]; };
layout(set = 0, binding = 2, scalar) readonly buffer ClusterVertices { uint cluster_vertices[]; };
layout(set = 0, binding = 3, scalar) readonly buffer ClusterTriangles { uint cluster_triangles[]; };
layout(set = 0, binding = 4, scalar) readonly buffer Positions { vec3 positions[]; };
layout(set = 0, binding = 5, scalar) readonly buffer Normals { vec3 normals[]; };
layout(set = 0, binding = 6, scalar) readonly buffer Meshes { Mesh meshes[]; };
layout(set = 0, binding = 7, scalar) readonly buffer Instances { Instance instances[]; };

// Work for the task shaders: 64 clusters of one instance from first. Pass
// 1's items are work[0, max_work_items), pass 2's the next max_work_items.
struct WorkItem {
    uint instance;
    uint first;  // Cluster number
};
layout(set = 0, binding = 8, scalar) buffer WorkItems {
    uint work_count[2];       // Per pass
    uint late_instance_count;
    uint late_cluster_count;
    WorkItem work[];
};

// What pass 1 found hidden behind last frame's depth, for pass 2 to test
// again: whole instances, and single clusters (instance, cluster).
layout(set = 0, binding = 14, scalar) buffer LateInstances { uint late_instances[]; };
layout(set = 0, binding = 15, scalar) buffer LateClusters { uvec2 late_clusters[]; };

// The depth pyramid: level 0 is half the screen, each texel the farthest
// (smallest, depth being reversed) depth of the pixels it covers.
layout(set = 0, binding = 16) uniform sampler2D hzb;

// Every cluster drawn this frame, numbered: the visibility buffer stores
// that number and a triangle. Clusters for the hardware rasterizer fill it
// from the front, those for the software rasterizer from the back.
layout(set = 0, binding = 9, scalar) buffer Visible {
    uint visible_count;
    uint sw_count;
    uint pad3, pad4;
    uvec2 visible[];  // (instance, cluster)
};

uint sw_slot(uint k) {
    return frame.max_visible - 1u - k;
}

// Per pixel: depth (reversed, as float bits) above the visible cluster
// number times 128 plus the triangle. An atomic max keeps the nearest, and
// 0 means nothing was drawn.
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
    uint clusters_occluded;   // In pass 1; some come back in pass 2
    uint clusters_late;       // Drawn in pass 2
    uint clusters_software;   // Drawn by the software rasterizer
} stats;

// Indirect arguments for each pass, written by args.comp: the cluster
// culling dispatch (x, y, z, and the number of work items), the mesh
// shader draw (x, y, z, and the first visible cluster it draws), and the
// software rasterizer's dispatch (x, y, z, and its first cluster).
layout(set = 0, binding = 12, scalar) buffer DrawArgs {
    uvec4 cull_args[2];
    uvec4 draw_args[2];
    uvec4 sw_args[2];
    uint pass_start[3];     // visible[pass_start[p], pass_start[p + 1]) is pass p's
    uint sw_pass_start[3];  // The same, counted from the back
};

vec3 to_world(Instance inst, vec3 p) {
    return vec3(dot(inst.rows[0].xyz, p) + inst.rows[0].w, dot(inst.rows[1].xyz, p) + inst.rows[1].w,
                dot(inst.rows[2].xyz, p) + inst.rows[2].w);
}

vec3 to_world_dir(Instance inst, vec3 v) {
    return vec3(dot(inst.rows[0].xyz, v), dot(inst.rows[1].xyz, v), dot(inst.rows[2].xyz, v));
}

bool sphere_in_frustum(vec3 c, float r) {
    bool inside = true;
    for (int k = 0; k < 5; ++k) inside = inside && dot(frame.cull_planes[k].xyz, c) + frame.cull_planes[k].w >= -r;
    return inside;
}

// How many pixels an error of world size `error` covers at the nearest
// point of a sphere. Clusters of one group compute this from identical
// numbers, so they always agree.
float projected_error(vec3 center, float radius, float error) {
    const float d = length(center - frame.cull_origin.xyz) - radius;
    return error * frame.lod_scale / max(d, frame.near_z);
}

// The screen rectangle a sphere covers, in [0, 1] texture coordinates,
// for a sphere in view space with z forward (Mara and McGuire 2013, as in
// zeux's niagara). False if the sphere reaches the near plane.
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

// Whether a world-space sphere is certainly hidden: its nearest point is
// farther than the farthest depth drawn anywhere in the rectangle it
// covers. Pass 1 asks from last frame's camera, against last frame's
// pyramid; pass 2 from this frame's.
bool occluded(vec3 center, float radius) {
    if ((frame.flags & flag_occlusion) == 0u) return false;
    if (push.pass == 0u && (frame.flags & flag_prev_valid) == 0u) return false;
    const mat4 view = push.pass == 0u ? frame.prev_view : frame.view;
    vec3 c = (view * vec4(center, 1.0)).xyz;
    c.z = -c.z;
    vec4 aabb;
    if (!project_sphere(c, radius, aabb)) return false;
    const vec2 screen = vec2(frame.width, frame.height);
    const vec2 lo = clamp(aabb.xy, 0.0, 1.0) * screen;
    const vec2 hi = clamp(aabb.zw, 0.0, 1.0) * screen - 0.5;
    // The finest level at which the rectangle spans at most 2x2 texels.
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

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
