// Shared by every shader of the web demo: the scene, laid out as
// renderer.js writes it, and the culling and LOD tests. A port of
// viewer/shaders/common.glsl.

struct Frame {
  view_proj: mat4x4f,
  inv_view_proj: mat4x4f,
  cull_planes: array<vec4f, 5>,  // World space: inside where dot(xyz, p) + w >= 0
  cull_origin: vec4f,
  origin: vec4f,
  width: u32,
  height: u32,
  instance_count: u32,
  flags: u32,
  lod_scale: f32,      // Pixels per unit of size at distance 1
  lod_threshold: f32,  // Allowed error on screen, in pixels
  near_z: f32,
  debug_mode: u32,
  max_work: u32,
  max_visible: u32,
  sw_max_pixels: f32,
  time: f32,
}

// include/geometry_file.hpp's gpu_cluster, as paged (include/paged_file.hpp):
// 112 bytes. `group` is its page, and the offsets are words into it.
struct Cluster {
  center: vec3f,
  radius: f32,
  cone_axis: vec3f,
  cone_cutoff: f32,
  lod_center: vec3f,
  lod_radius: f32,
  parent_center: vec3f,
  parent_radius: f32,
  lod_error: f32,
  parent_error: f32,
  vertex_offset: u32,
  triangle_offset: u32,
  vertex_count: u32,
  triangle_count: u32,
  level: u32,
  group: u32,
  creator: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

struct Mesh {
  first_cluster: u32,
  cluster_count: u32,
  pad0: u32,
  pad1: u32,
  bounds: vec4f,
  lod_bounds: vec4f,
}

struct Instance {
  rows: array<vec4f, 3>,  // The 3x4 to-world transform, by rows
  mesh: u32,
  scale: f32,
  pad0: u32,
  pad1: u32,
}

const FLAG_CONE = 1u;
const FLAG_FRUSTUM = 2u;
const FLAG_SOFTWARE = 4u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
// Every page is loaded on the web: the page table holds each one's first
// word in the pool.
@group(0) @binding(2) var<storage, read> page_table: array<u32>;
@group(0) @binding(3) var<storage, read> pool: array<u32>;
@group(0) @binding(6) var<storage, read> meshes: array<Mesh>;
@group(0) @binding(7) var<storage, read> instances: array<Instance>;

fn to_world(inst: Instance, p: vec3f) -> vec3f {
  return vec3f(dot(inst.rows[0].xyz, p) + inst.rows[0].w, dot(inst.rows[1].xyz, p) + inst.rows[1].w,
               dot(inst.rows[2].xyz, p) + inst.rows[2].w);
}

fn to_world_dir(inst: Instance, v: vec3f) -> vec3f {
  return vec3f(dot(inst.rows[0].xyz, v), dot(inst.rows[1].xyz, v), dot(inst.rows[2].xyz, v));
}

fn cluster_position(c: Cluster, k: u32) -> vec3f {
  let at = page_table[c.group] + c.vertex_offset + 4u * k;
  return vec3f(bitcast<f32>(pool[at]), bitcast<f32>(pool[at + 1u]), bitcast<f32>(pool[at + 2u]));
}

// Octahedral, two snorm16.
fn cluster_normal(c: Cluster, k: u32) -> vec3f {
  let packed = pool[page_table[c.group] + c.vertex_offset + 4u * k + 3u];
  let e = vec2f(f32((i32(packed) << 16u) >> 16u), f32(i32(packed) >> 16u)) / 32767.0;
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0) {
    n = vec3f((1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0), (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0), n.z);
  }
  return normalize(n);
}

fn cluster_triangle(c: Cluster, t: u32) -> u32 {
  return pool[page_table[c.group] + c.triangle_offset + t];
}

fn sphere_in_frustum(c: vec3f, r: f32) -> bool {
  var inside = true;
  for (var k = 0; k < 5; k++) {
    inside = inside && dot(frame.cull_planes[k].xyz, c) + frame.cull_planes[k].w >= -r;
  }
  return inside;
}

// How many pixels an error of world size `error` covers at the nearest
// point of a sphere. Clusters of one group compute this from identical
// numbers, so they always agree.
fn projected_error(center: vec3f, radius: f32, error: f32) -> f32 {
  let d = length(center - frame.cull_origin.xyz) - radius;
  return error * frame.lod_scale / max(d, frame.near_z);
}

fn hash(x0: u32) -> u32 {
  var x = x0;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}
