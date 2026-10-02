// shared by every web shader: the scene as renderer.js writes it, culling and
// lod tests. port of common.glsl.

struct Frame {
  view_proj: mat4x4f,
  inv_view_proj: mat4x4f,
  cull_planes: array<vec4f, 5>,  // world space: inside where dot(xyz, p) + w >= 0
  cull_origin: vec4f,
  origin: vec4f,
  width: u32,
  height: u32,
  instance_count: u32,
  flags: u32,
  lod_scale: f32,      // pixels per unit at distance 1
  lod_threshold: f32,  // allowed error on screen, pixels
  near_z: f32,
  debug_mode: u32,
  max_work: u32,
  max_visible: u32,
  sw_max_pixels: f32,
  time: f32,
  frame_index: u32,   // from 1: stamps pages used and requested
  max_requests: u32,
  pad0: u32,
  pad1: u32,
  unused0: mat4x4f,  // was the old shadow map's camera; kept for the layout
  unused1: f32,
  unused2: f32,
  pad2: u32,
  pad3: u32,
  // occlusion: the pyramid's cameras. pass 1 against last frame's pyramid and
  // camera; pass 2 this frame's, built after pass 1.
  view: mat4x4f,
  prev_view: mat4x4f,
  p00: f32,  // projection scale, x and y
  p11: f32,
  hzb_levels: u32,
  pad4: u32,
  // taa: last frame's unjittered camera, this frame's offset (clip space),
  // whether last frame's image fits.
  prev_view_proj: mat4x4f,
  jitter: vec2f,
  taa_valid: u32,
  prev_time: f32,  // last frame's time, for motion
}

// gpu_cluster (include/geometry_file.hpp) as paged: 112 bytes. `group` is its
// page; offsets are words into it.
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
  // corner on the model's grid. three u32, not vec3u, which would align to 16
  // and make 128 bytes.
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
  grid: vec4f,  // position snapping: xyz grid point 0, w step
}

struct Instance {
  rows: array<vec4f, 3>,  // 3x4 to-world, by rows
  mesh: u32,
  scale: f32,
  material: u32,  // into compute.wgsl's materials
  anim: u32,      // 0: still. else moving (animate()): phase in low 8 bits, bit 8 reverses
}

const FLAG_CONE = 1u;
const FLAG_FRUSTUM = 2u;
const FLAG_SOFTWARE = 4u;
const FLAG_SHADOWS = 16u;
const FLAG_OCCLUSION = 32u;
const FLAG_PREV_VALID = 64u;
const FLAG_AO = 512u;  // ambient occlusion (ao.wgsl)
const FLAG_SOFT_SHADOWS = 1024u;  // contact-hardening penumbras (sunlight())  // last frame's pyramid fits this frame

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
// pages stream into the pool (web/streamer.js); the page table holds each one's
// first word, or NO_PAGE.
const NO_PAGE = 0xffffffffu;
@group(0) @binding(2) var<storage, read> page_table: array<u32>;
@group(0) @binding(3) var<storage, read> pool: array<u32>;
@group(0) @binding(6) var<storage, read> meshes: array<Mesh>;
@group(0) @binding(7) var<storage, read> instances: array<Instance>;

// as common.glsl: a moving instance turns on the spot and drifts round a small
// circle, from the time. its last transform (load_prev_instance) is where last
// frame's pyramid and image have it.
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

// inverse of to_world: rotation and uniform scale.
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

// a cluster's vertices and triangles, bit-packed (include/paged_file.hpp): a vertex is its offsets
// from the cluster's corner in bx, by, bz bits and an 11 + 11 bit octahedral normal; a triangle
// three ib-bit indices. the widths ride in the level word.
fn cluster_level(c: Cluster) -> u32 { return c.level & 255u; }

fn cluster_widths(c: Cluster) -> vec4u {  // bx, by, bz, ib
  return (vec4u(c.level) >> vec4u(8u, 12u, 16u, 20u)) & vec4u(15u);
}

// `count` bits (at most 31) at a bit offset into the run starting at word `at`.
fn read_bits(at: u32, bit: u32, count: u32) -> u32 {
  let w = at + (bit >> 5u);
  let s = bit & 31u;
  var v = pool[w] >> s;
  if (s + count > 32u) { v |= pool[w + 1u] << (32u - s); }
  return v & ((1u << count) - 1u);
}

fn cluster_position(c: Cluster, grid: vec4f, k: u32) -> vec3f {
  let b = cluster_widths(c);
  let at = page_table[c.group] + c.vertex_offset;
  let bit = k * (b.x + b.y + b.z + 22u);
  let d = vec3u(read_bits(at, bit, b.x), read_bits(at, bit + b.x, b.y), read_bits(at, bit + b.x + b.y, b.z));
  return grid.xyz + grid.w * vec3f(vec3u(c.origin_x, c.origin_y, c.origin_z) + d);
}

fn cluster_normal(c: Cluster, k: u32) -> vec3f {
  let b = cluster_widths(c);
  let xyz = b.x + b.y + b.z;
  let uv = read_bits(page_table[c.group] + c.vertex_offset, k * (xyz + 22u) + xyz, 22u);
  let e = vec2f(f32(uv & 2047u), f32(uv >> 11u)) / 2047.0 * 2.0 - 1.0;
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0) {
    n = vec3f((1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0), (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0), n.z);
  }
  return normalize(n);
}

// triangle t as a | b << 8 | c << 16.
fn cluster_triangle(c: Cluster, t: u32) -> u32 {
  let ib = cluster_widths(c).w;
  let mask = (1u << ib) - 1u;
  let v = read_bits(page_table[c.group] + c.triangle_offset, t * 3u * ib, 3u * ib);
  return (v & mask) | (((v >> ib) & mask) << 8u) | (((v >> (2u * ib)) & mask) << 16u);
}

fn sphere_in_frustum(c: vec3f, r: f32) -> bool {
  var inside = true;
  for (var k = 0; k < 5; k++) {
    inside = inside && dot(frame.cull_planes[k].xyz, c) + frame.cull_planes[k].w >= -r;
  }
  return inside;
}

// pixels an error of world size `error` covers at a sphere's nearest point. a
// group computes it from identical numbers, so it agrees.
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
