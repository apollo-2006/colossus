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
    uint pad0, pad1;
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
} frame;

const uint flag_cone_culling = 1u;
const uint flag_frustum_culling = 2u;
const uint flag_wireframe = 4u;

layout(set = 0, binding = 1, scalar) readonly buffer Clusters { Cluster clusters[]; };
layout(set = 0, binding = 2, scalar) readonly buffer ClusterVertices { uint cluster_vertices[]; };
layout(set = 0, binding = 3, scalar) readonly buffer ClusterTriangles { uint cluster_triangles[]; };
layout(set = 0, binding = 4, scalar) readonly buffer Positions { vec3 positions[]; };
layout(set = 0, binding = 5, scalar) readonly buffer Normals { vec3 normals[]; };
layout(set = 0, binding = 6, scalar) readonly buffer Meshes { Mesh meshes[]; };
layout(set = 0, binding = 7, scalar) readonly buffer Instances { Instance instances[]; };

// Work for the task shaders: 64 clusters of one instance from first.
struct WorkItem {
    uint instance;
    uint first;  // Cluster number
};
layout(set = 0, binding = 8, scalar) buffer WorkItems {
    uint work_count;
    uint pad0, pad1, pad2;
    WorkItem work[];
};

// Every cluster drawn this frame, numbered: the visibility buffer stores
// that number and a triangle.
layout(set = 0, binding = 9, scalar) buffer Visible {
    uint visible_count;
    uint pad3, pad4, pad5;
    uvec2 visible[];  // (instance, cluster)
};

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
    uint pad6;
} stats;

// The task shader's indirect dispatch, written by args.comp.
layout(set = 0, binding = 12, scalar) buffer DrawArgs {
    uint task_x, task_y, task_z;
} draw_args;

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

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
