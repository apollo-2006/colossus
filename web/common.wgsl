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
  frame_index: u32,   // Counts from 1: stamps pages used and requested
  max_requests: u32,
  pad0: u32,
  pad1: u32,
  shadow_view_proj: mat4x4f,  // World to the shadow map's clip space
  // Over 0 for the shadow pass: the sun's camera is orthographic, and an
  // error covers error * ortho_scale shadow map texels wherever it is.
  ortho_scale: f32,
  shadow_texel: f32,  // A shadow map texel's size in world units
  pad2: u32,
  pad3: u32,
  // Occlusion culling: the depth pyramid's cameras. Pass 1 tests against
  // last frame's pyramid from last frame's camera; pass 2 against this
  // frame's, built after pass 1.
  view: mat4x4f,
  prev_view: mat4x4f,
  p00: f32,  // Projection scale in x and y
  p11: f32,
  hzb_levels: u32,
  pad4: u32,
  // Temporal antialiasing: last frame's camera without its sub-pixel
  // offset, this frame's offset (clip space), and whether last frame's
  // image fits this one.
  prev_view_proj: mat4x4f,
  jitter: vec2f,
  taa_valid: u32,
  prev_time: f32,  // Last frame's time, for moving instances
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
  // Its corner on the model's grid. Three u32, not a vec3u: that would
  // align to 16 bytes and make the struct 128 bytes, not the file's 112.
  origin_x: u32,
  origin_y: u32,
  origin_z: u32,
}

struct Mesh {
  first_cluster: u32,
  cluster_count: u32,
  pad0: u32,
  pad1: u32,
  bounds: vec4f,
  lod_bounds: vec4f,
  grid: vec4f,  // Where positions snap: xyz grid point 0, w the step
}

struct Instance {
  rows: array<vec4f, 3>,  // The 3x4 to-world transform, by rows
  mesh: u32,
  scale: f32,
  material: u32,  // Into compute.wgsl's materials
  anim: u32,      // 0: still. Else moving (see animate()): phase in the low 8 bits, bit 8 turns it the other way
}

const FLAG_CONE = 1u;
const FLAG_FRUSTUM = 2u;
const FLAG_SOFTWARE = 4u;
const FLAG_SHADOW_PASS = 8u;  // Culling and drawing for the shadow map
const FLAG_SHADOWS = 16u;
const FLAG_OCCLUSION = 32u;
const FLAG_PREV_VALID = 64u;  // Last frame's pyramid fits this frame

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
// Pages stream into the pool (web/streamer.js): the page table holds each
// one's first word there, or NO_PAGE.
const NO_PAGE = 0xffffffffu;
@group(0) @binding(2) var<storage, read> page_table: array<u32>;
@group(0) @binding(3) var<storage, read> pool: array<u32>;
@group(0) @binding(6) var<storage, read> meshes: array<Mesh>;
@group(0) @binding(7) var<storage, read> instances: array<Instance>;

// As in the viewer's common.glsl: a moving instance turns on the spot and
// drifts round a small circle, computed from the time. Last frame's
// transform (load_prev_instance) is where last frame's depth pyramid and
// image have it.
fn animate(inst_in: Instance, t: f32) -> Instance {
  var inst = inst_in;
  if (inst.anim == 0u) { return inst; }
  let phase = f32(inst.anim & 255u) * (6.2831853 / 256.0);
  let turn = phase + t * select(0.7, -0.7, (inst.anim & 256u) != 0u);
  let c = cos(turn);
  let s = sin(turn);
  let drift = vec2f(cos(phase + t * 0.9), sin(phase + t * 0.9)) * 0.2;
  let r0 = inst.rows[0].xyz;
  let r2 = inst.rows[2].xyz;
  inst.rows[0] = vec4f(c * r0 + s * r2, inst.rows[0].w + drift.x);
  inst.rows[2] = vec4f(-s * r0 + c * r2, inst.rows[2].w + drift.y);
  return inst;
}

fn load_instance(i: u32) -> Instance {
  return animate(instances[i], frame.time);
}

fn load_prev_instance(i: u32) -> Instance {
  return animate(instances[i], frame.prev_time);
}

// The inverse of to_world: a rotation and a uniform scale.
fn from_world(inst: Instance, p: vec3f) -> vec3f {
  let d = p - vec3f(inst.rows[0].w, inst.rows[1].w, inst.rows[2].w);
  let c0 = vec3f(inst.rows[0].x, inst.rows[1].x, inst.rows[2].x);
  let c1 = vec3f(inst.rows[0].y, inst.rows[1].y, inst.rows[2].y);
  let c2 = vec3f(inst.rows[0].z, inst.rows[1].z, inst.rows[2].z);
  return vec3f(dot(c0, d), dot(c1, d), dot(c2, d)) / (inst.scale * inst.scale);
}

fn to_world(inst: Instance, p: vec3f) -> vec3f {
  return vec3f(dot(inst.rows[0].xyz, p) + inst.rows[0].w, dot(inst.rows[1].xyz, p) + inst.rows[1].w,
               dot(inst.rows[2].xyz, p) + inst.rows[2].w);
}

fn to_world_dir(inst: Instance, v: vec3f) -> vec3f {
  return vec3f(dot(inst.rows[0].xyz, v), dot(inst.rows[1].xyz, v), dot(inst.rows[2].xyz, v));
}

// A vertex is two words: 14-bit offsets from the cluster's corner on the
// model's grid, and an 11 + 11 bit octahedral normal (include/paged_file.hpp).
fn cluster_position(c: Cluster, grid: vec4f, k: u32) -> vec3f {
  let at = page_table[c.group] + c.vertex_offset + 2u * k;
  let w0 = pool[at];
  let w1 = pool[at + 1u];
  let d = vec3u(w0 & 0x3fffu, (w0 >> 14u) & 0x3fffu, (w0 >> 28u) | ((w1 & 0x3ffu) << 4u));
  return grid.xyz + grid.w * vec3f(vec3u(c.origin_x, c.origin_y, c.origin_z) + d);
}

fn cluster_normal(c: Cluster, k: u32) -> vec3f {
  let w1 = pool[page_table[c.group] + c.vertex_offset + 2u * k + 1u];
  let e = vec2f(f32((w1 >> 10u) & 2047u), f32(w1 >> 21u)) / 2047.0 * 2.0 - 1.0;
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0) {
    n = vec3f((1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0), (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0), n.z);
  }
  return normalize(n);
}

// Triangles are three bytes each, packed end to end.
fn cluster_triangle(c: Cluster, t: u32) -> u32 {
  let byte = 3u * t;
  let at = page_table[c.group] + c.triangle_offset + byte / 4u;
  let shift = (byte % 4u) * 8u;
  var v = pool[at] >> shift;
  if (shift > 8u) { v |= pool[at + 1u] << (32u - shift); }
  return v & 0xffffffu;
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
  if (frame.ortho_scale > 0.0) { return error * frame.ortho_scale; }
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
