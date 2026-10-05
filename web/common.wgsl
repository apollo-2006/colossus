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
  page_count: u32,  // page_table's entries; the pages' shared bounds follow
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

// gpu_cluster (include/geometry_file.hpp) as paged, unpacked by load_cluster().
// `group` is its page; offsets are words into it.
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

// as stored: packed_cluster (include/paged_file.hpp), 48 bytes.
struct PackedCluster {
  center: vec3f,
  radius: f32,
  cone: u32,     // axis u (11) | v (11) << 11 | cutoff (10) << 22
  offsets: u32,  // vertex_offset | triangle_offset << 16
  level: u32,    // level word | vertex_count << 24
  group: u32,
  creator: u32,
  origin_x: u32,  // 24 bits each, origin_x | triangle_count << 24
  origin_y: u32,
  origin_z: u32,
}

struct Mesh {
  first_cluster: u32,
  cluster_count: u32,
  pose_joints: u32,  // every skinned slot's joints, in all models: where last frame's start
  pad1: u32,
  bounds: vec4f,
  lod_bounds: vec4f,
  grid: vec4f,  // position snapping: xyz grid point 0, w step
  tex: vec4u,   // texture tiles: first page, levels (0: none), width, height (compute.wgsl)
  skin: vec4u,  // skinned: joints (0: none), first pose slot, its first joint, anchor joint
}

struct Instance {
  rows: array<vec4f, 3>,  // 3x4 to-world, by rows
  mesh: u32,
  scale: f32,
  material: u32,  // into compute.wgsl's materials
  anim: u32,      // 0: still. else moving (animate()): phase in low 8 bits, bit 8 reverses;
                  // bit 10 sways (deform()); bit 11 skinned, its pose slot from bit 16
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
@group(0) @binding(1) var<storage, read> packed_clusters: array<PackedCluster>;
// pages stream into the pool (web/streamer.js); the page table holds each one's
// first word, or NO_PAGE. after its frame.page_count entries, each page's shared
// bounds: centre, radius, error.
const NO_PAGE = 0xffffffffu;
@group(0) @binding(2) var<storage, read> page_table: array<u32>;

fn decode_octahedral(uv: u32) -> vec3f {
  let e = vec2f(f32(uv & 2047u), f32((uv >> 11u) & 2047u)) / 2047.0 * 2.0 - 1.0;
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0) {
    n = vec3f((1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0), (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0), n.z);
  }
  return normalize(n);
}

fn page_sphere(p: u32) -> vec4f {
  let at = frame.page_count + 5u * p;
  return bitcast<vec4f>(vec4u(page_table[at], page_table[at + 1u], page_table[at + 2u], page_table[at + 3u]));
}

fn page_error(p: u32) -> f32 { return bitcast<f32>(page_table[frame.page_count + 5u * p + 4u]); }

// for the binary searches over a model's clusters (sorted by it).
fn cluster_parent_error(i: u32) -> f32 { return page_error(packed_clusters[i].group); }

fn load_cluster(i: u32) -> Cluster {
  let p = packed_clusters[i];
  var c: Cluster;
  c.center = p.center;
  c.radius = p.radius;
  let steps = p.cone >> 22u;
  c.cone_axis = select(decode_octahedral(p.cone), vec3f(0.0, 0.0, 1.0), steps == 1023u);
  c.cone_cutoff = f32(steps) / 1023.0 * 2.0 - 1.0;
  c.vertex_offset = p.offsets & 0xffffu;
  c.triangle_offset = p.offsets >> 16u;
  c.level = p.level & 0xffffffu;
  c.vertex_count = p.level >> 24u;
  c.group = p.group;
  c.creator = p.creator;
  c.origin_x = p.origin_x & 0xffffffu;
  c.origin_y = p.origin_y;
  c.origin_z = p.origin_z;
  c.triangle_count = p.origin_x >> 24u;
  let parent = page_sphere(p.group);
  c.parent_center = parent.xyz;
  c.parent_radius = parent.w;
  c.parent_error = page_error(p.group);
  if (p.creator == NO_PAGE) {
    c.lod_center = p.center;
    c.lod_radius = p.radius;
    c.lod_error = 0.0;
  } else {
    let lod = page_sphere(p.creator);
    c.lod_center = lod.xyz;
    c.lod_radius = lod.w;
    c.lod_error = page_error(p.creator);
  }
  return c;
}
@group(0) @binding(3) var<storage, read> pool: array<u32>;
@group(0) @binding(6) var<storage, read> meshes: array<Mesh>;
@group(0) @binding(7) var<storage, read> instances: array<Instance>;

// as common.glsl: a moving instance turns on the spot and drifts round a small
// circle, from the time. its last transform (load_prev_instance) is where last
// frame's pyramid and image have it.
fn animate(inst_in: Instance, t: f32) -> Instance {
  var inst = inst_in;
  if ((inst.anim & 512u) == 0u) { return inst; }
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
  return (vec4u(c.level) >> vec4u(8u, 12u, 16u, 20u)) & vec4u(15u, 15u, 15u, 7u);
}

// `count` bits (at most 31) at a bit offset into the run starting at word `at`.
fn read_bits(at: u32, bit: u32, count: u32) -> u32 {
  let w = at + (bit >> 5u);
  let s = bit & 31u;
  var v = pool[w] >> s;
  if (s + count > 32u) { v |= pool[w + 1u] << (32u - s); }
  return v & ((1u << count) - 1u);
}

// deforming instances (anim bit 10) sway, as common.glsl's deform(): a bend growing with
// the square of the height and a ripple up it, sideways, functions of height alone (crack
// free, exactly invertible). slope <= 0.128: lengths stretch <= 1.066, points move <= 0.1 r.
const DEFORM_A = 0.08;
const DEFORM_B = 0.02;
const DEFORM_K = 9.0;
const DEFORM_REACH = 0.1;
const DEFORM_STRETCH = 1.07;

fn deforming(inst: Instance) -> bool { return (inst.anim & 1024u) != 0u; }

// the sideways offset at model height y and its slope along the height.
struct DeformOffset {
  offset: vec3f,
  slope: vec3f,
}

fn deform_offset(inst: Instance, bounds: vec4f, y: f32, t: f32) -> DeformOffset {
  let r = bounds.w;
  let h = clamp((y - (bounds.y - r)) / (2.0 * r), 0.0, 1.0);
  let phase = f32(inst.anim & 255u) * (6.2831853 / 256.0);
  let heading = phase * 3.0;
  let dir = vec3f(cos(heading), 0.0, sin(heading));
  let perp = vec3f(-dir.z, 0.0, dir.x);
  let sway = sin(1.3 * t + phase);
  let arg = DEFORM_K * h - 2.2 * t + phase;
  var d: DeformOffset;
  d.slope = dir * (DEFORM_A * h * sway) + perp * (DEFORM_B * (sin(arg) + h * DEFORM_K * cos(arg)) * 0.5);
  d.offset = dir * (DEFORM_A * r * h * h * sway) + perp * (DEFORM_B * r * h * sin(arg));
  return d;
}

fn deform(inst: Instance, bounds: vec4f, p: vec3f, t: f32) -> vec3f {
  if (!deforming(inst)) { return p; }
  return p + deform_offset(inst, bounds, p.y, t).offset;
}

fn undeform(inst: Instance, bounds: vec4f, p: vec3f, t: f32) -> vec3f {
  if (!deforming(inst)) { return p; }
  return p - deform_offset(inst, bounds, p.y, t).offset;
}

// a gradient fixed to the rest surface, at drawn point p: the shear's inverse transpose.
fn deform_gradient(inst: Instance, bounds: vec4f, p: vec3f, g: vec3f, t: f32) -> vec3f {
  if (!deforming(inst)) { return g; }
  let slope = deform_offset(inst, bounds, p.y, t).slope;
  return g - vec3f(0.0, dot(slope, g), 0.0);
}

fn deform_normal(inst: Instance, bounds: vec4f, p: vec3f, n: vec3f, t: f32) -> vec3f {
  if (!deforming(inst)) { return n; }
  return normalize(deform_gradient(inst, bounds, p, n, t));
}

fn deform_reach_of(inst: Instance) -> f32 { return select(0.0, DEFORM_REACH * meshes[inst.mesh].bounds.w, deforming(inst)); }
fn deform_stretch_of(inst: Instance) -> f32 { return select(1.0, DEFORM_STRETCH, deforming(inst)); }

// skinning (include/skeleton.hpp, viewer/shaders/common.glsl, which this follows). each skinned
// instance follows a pose slot, posed on the cpu each frame. the page table buffer carries,
// after the table (page_count words) and the pages' shared bounds (5 each), each page's
// joints (2 words, a bit each), then this frame's joints (16 words each: three rows and |A -
// I|, |A - A_anchor|), last frame's, and per slot how far its joints carry the model's
// bounds and lod bounds from the anchor, and the fastest point.
struct Joint {
  r0: vec4f,
  r1: vec4f,
  r2: vec4f,
  norms: vec4f,  // x: |A - I|, y: |A - A_anchor| (spectral)
}

// false compiles skinning out (the renderer's second set of pipelines, used while nothing
// placed is skinned): its untaken branches still cost registers.
override SKINNING: bool = true;
fn skinned(inst: Instance) -> bool { return SKINNING && (inst.anim & 2048u) != 0u; }
fn pose_slot(inst: Instance) -> u32 { return inst.anim >> 16u; }
fn joint_base(inst: Instance) -> u32 {
  let sk = meshes[inst.mesh].skin;
  return sk.z + (pose_slot(inst) - sk.y) * sk.x;
}
fn pose_vec4(at: u32) -> vec4f {
  return bitcast<vec4f>(vec4u(page_table[at], page_table[at + 1u], page_table[at + 2u], page_table[at + 3u]));
}
fn load_joint(inst: Instance, j: u32, prev: bool) -> Joint {
  let at = 8u * frame.page_count + select(0u, 16u * meshes[inst.mesh].pose_joints, prev) + 16u * j;
  return Joint(pose_vec4(at), pose_vec4(at + 4u), pose_vec4(at + 8u), pose_vec4(at + 12u));
}
fn pose_info(inst: Instance) -> vec4f {
  return pose_vec4(8u * frame.page_count + 32u * meshes[inst.mesh].pose_joints + 4u * pose_slot(inst));
}
fn page_joints(page: u32) -> vec2u {
  let at = 6u * frame.page_count + 2u * page;
  return vec2u(page_table[at], page_table[at + 1u]);
}
fn joint_point(j: Joint, p: vec3f) -> vec3f {
  let q = vec4f(p, 1.0);
  return vec3f(dot(j.r0, q), dot(j.r1, q), dot(j.r2, q));
}
fn joint_dir(j: Joint, d: vec3f) -> vec3f { return vec3f(dot(j.r0.xyz, d), dot(j.r1.xyz, d), dot(j.r2.xyz, d)); }

// a sphere around a skinned sphere's points, nested level to level for the lod test: centred on
// the anchor joint's image of the centre, grown by how far each of the page's joints carries
// points from where the anchor does (lod_sphere() in common.glsl explains).
fn lod_sphere(inst: Instance, page: u32, center: vec3f, radius: f32) -> vec4f {
  if (!skinned(inst)) { return vec4f(center, radius + deform_reach_of(inst)); }
  let at = joint_base(inst);
  let a = load_joint(inst, at + meshes[inst.mesh].skin.w, false);
  let c = joint_point(a, center);
  let bits = page_joints(page);
  var grow = 0.0;
  for (var h = 0u; h < 2u; h++) {
    var b = select(bits.y, bits.x, h == 0u);
    while (b != 0u) {
      let j = firstTrailingBit(b) + 32u * h;
      b &= b - 1u;
      let jt = load_joint(inst, at + j, false);
      grow = max(grow, length(joint_point(jt, center) - c) + jt.norms.y * radius);
    }
  }
  return vec4f(c, (1.0 + a.norms.x) * radius + grow);
}

// the other nested sphere: the rest sphere grown by how far the page's joints move its points.
fn rest_sphere(inst: Instance, page: u32, center: vec3f, radius: f32) -> vec4f {
  let at = joint_base(inst);
  let bits = page_joints(page);
  var reach = 0.0;
  for (var h = 0u; h < 2u; h++) {
    var b = select(bits.y, bits.x, h == 0u);
    while (b != 0u) {
      let j = firstTrailingBit(b) + 32u * h;
      b &= b - 1u;
      let jt = load_joint(inst, at + j, false);
      reach = max(reach, length(joint_point(jt, center) - center) + jt.norms.x * radius);
    }
  }
  return vec4f(center, radius + reach);
}

// a sphere around where a cluster is drawn, for culling only: on its lowest joint's image.
fn drawn_sphere(inst: Instance, page: u32, center: vec3f, radius: f32) -> vec4f {
  if (!skinned(inst)) { return vec4f(center, radius + deform_reach_of(inst)); }
  let at = joint_base(inst);
  let bits = page_joints(page);
  if (all(bits == vec2u(0u))) { return vec4f(center, radius); }
  let first = select(32u + firstTrailingBit(bits.y), firstTrailingBit(bits.x), bits.x != 0u);
  let a = load_joint(inst, at + first, false);
  let c = joint_point(a, center);
  var grow = 0.0;
  for (var h = 0u; h < 2u; h++) {
    var b = select(bits.y, bits.x, h == 0u);
    while (b != 0u) {
      let j = firstTrailingBit(b) + 32u * h;
      b &= b - 1u;
      let jt = load_joint(inst, at + j, false);
      let d0 = jt.r0.xyz - a.r0.xyz;
      let d1 = jt.r1.xyz - a.r1.xyz;
      let d2 = jt.r2.xyz - a.r2.xyz;
      grow = max(grow, length(joint_point(jt, center) - c) + sqrt(dot(d0, d0) + dot(d1, d1) + dot(d2, d2)) * radius);
    }
  }
  return vec4f(c, (1.0 + a.norms.x) * radius + grow);
}

// a model's whole sphere (bounds) or its sphere around every lod sphere (lod_bounds).
fn instance_sphere(inst: Instance, sphere: vec4f, lod: bool) -> vec4f {
  if (!skinned(inst)) { return vec4f(sphere.xyz, sphere.w + deform_reach_of(inst)); }
  let a = load_joint(inst, joint_base(inst) + meshes[inst.mesh].skin.w, false);
  let info = pose_info(inst);
  return vec4f(joint_point(a, sphere.xyz), (1.0 + a.norms.x) * sphere.w + select(info.x, info.y, lod));
}

// errors as drawn: a swaying instance's stretched; a skinned one's already hold its
// animations (measured posed when built).
fn grown_error(inst: Instance, error: f32) -> f32 { return select(error * deform_stretch_of(inst), error, skinned(inst)); }
fn error_floor(inst: Instance, limit: f32) -> f32 { return select(limit / deform_stretch_of(inst), limit, skinned(inst)); }

// words from a cluster's vertex run to its skin: past its texture coordinates if any.
fn skin_run(c: Cluster) -> u32 {
  let b = cluster_widths(c);
  let at = (c.vertex_count * (b.x + b.y + b.z + 22u) + 31u) / 32u + 1u;
  if ((c.level & 0x800000u) == 0u) { return at; }
  let widths = pool[page_table[c.group] + c.vertex_offset + at + 1u];
  let bu = min(widths & 31u, 16u);
  let bv = min((widths >> 5u) & 31u, 16u);
  return at + 2u + (c.vertex_count * (bu + bv) + 31u) / 32u;
}

// vertex k skinned, now or as last frame; a direction turned by its joints.
fn skin_point(inst: Instance, c: Cluster, k: u32, p: vec3f, prev: bool) -> vec3f {
  let run = page_table[c.group] + c.vertex_offset + skin_run(c) + 2u * k;
  let js = pool[run];
  let ws = pool[run + 1u];
  let at = joint_base(inst);
  let last = meshes[inst.mesh].skin.x - 1u;
  var out_p = vec3f(0.0);
  for (var i = 0u; i < 4u; i++) {
    let w = f32((ws >> (8u * i)) & 255u) / 255.0;
    if (w == 0.0) { continue; }
    out_p += w * joint_point(load_joint(inst, at + min((js >> (8u * i)) & 255u, last), prev), p);
  }
  return out_p;
}

fn skin_dir(inst: Instance, c: Cluster, k: u32, d: vec3f) -> vec3f {
  let run = page_table[c.group] + c.vertex_offset + skin_run(c) + 2u * k;
  let js = pool[run];
  let ws = pool[run + 1u];
  let at = joint_base(inst);
  let last = meshes[inst.mesh].skin.x - 1u;
  var out_d = vec3f(0.0);
  for (var i = 0u; i < 4u; i++) {
    let w = f32((ws >> (8u * i)) & 255u) / 255.0;
    if (w != 0.0) { out_d += w * joint_dir(load_joint(inst, at + min((js >> (8u * i)) & 255u, last), false), d); }
  }
  return out_d;
}

// vertex k of a cluster as drawn now (model space): skinned, or swaying, or as stored.
fn drawn_vertex(inst: Instance, c: Cluster, k: u32) -> vec3f {
  let m = meshes[inst.mesh];
  let p = cluster_position(c, m.grid, k);
  if (skinned(inst)) { return skin_point(inst, c, k, p, false); }
  return deform(inst, m.bounds, p, frame.time);
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
  return decode_octahedral(read_bits(page_table[c.group] + c.vertex_offset, k * (xyz + 22u) + xyz, 22u));
}

// a textured cluster's (level bit 23) texture coordinates follow its vertex run and the word
// after it: corner u0 | v0 << 16 on a 65535-step grid, widths bu | bv << 5, then per vertex its
// offsets (include/paged_file.hpp). widths past 16 are corrupt: capped.
fn cluster_uv(c: Cluster, k: u32) -> vec2f {
  let b = cluster_widths(c);
  let run = page_table[c.group] + c.vertex_offset + (c.vertex_count * (b.x + b.y + b.z + 22u) + 31u) / 32u + 1u;
  let corner = pool[run];
  let widths = pool[run + 1u];
  let bu = min(widths & 31u, 16u);
  let bv = min((widths >> 5u) & 31u, 16u);
  let bit = 64u + k * (bu + bv);
  return vec2f(f32((corner & 0xffffu) + read_bits(run, bit, bu)), f32((corner >> 16u) + read_bits(run, bit + bu, bv))) / 65535.0;
}

fn cluster_textured(c: Cluster) -> bool { return (c.level & 0x800000u) != 0u; }

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

// the most pixels an error of world size `error` can move anywhere in a sphere, as
// common.glsl's: lod_scale * e * sec / z, z >= depth - radius, sec of the sphere's
// farthest angle off axis, no more than the screen corner's.
fn projected_sec(v: vec3f, radius: f32) -> f32 {
  let dist = length(v);
  let corner = sqrt(1.0 + f32(frame.width * frame.width + frame.height * frame.height) / (4.0 * frame.lod_scale * frame.lod_scale));
  if (radius >= dist) { return corner; }
  let c = dot(v, frame.cull_planes[4].xyz) / dist;
  let s = sqrt(max(1.0 - c * c, 0.0));
  let sa = radius / dist;
  let cos_far = c * sqrt(1.0 - sa * sa) - s * sa;  // cos(centre angle + angular radius)
  return select(1.0 / cos_far, corner, cos_far <= 1.0 / corner);
}

fn projected_error(center: vec3f, radius: f32, error: f32) -> f32 {
  let v = center - frame.cull_origin.xyz;
  let z = max(dot(v, frame.cull_planes[4].xyz) - radius, frame.near_z);  // plane 4: near, facing forward
  return error * frame.lod_scale * projected_sec(v, radius) / z;
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
