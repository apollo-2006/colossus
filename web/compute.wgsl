// the web demo's compute passes: instance and cluster culling, the software
// rasterizer, shading. the vulkan viewer's, shaped by what webgpu lacks:
//
// * no mesh shaders: big clusters go through raster.wgsl, a render pipeline
//   pulling triangles from storage.
// * no 64-bit atomics: the software rasterizer runs twice, first the nearest
//   depth by 32-bit atomic max, then the triangle wherever its depth won.
// * no ray queries: shadows from virtual shadow maps (vsm.wgsl).
//
// hardware and software results stay apart; shade() takes the nearer.

struct Counters {
  work: array<atomic<u32>, 2>,  // work items, per pass
  hw: atomic<u32>,
  sw: atomic<u32>,
  instances_visible: atomic<u32>,
  clusters_tested: atomic<u32>,
  triangles: atomic<u32>,
  overflow: atomic<u32>,
  late_instances: atomic<u32>,  // hidden in pass 1, for pass 2
  late_clusters: atomic<u32>,
  instances_occluded: atomic<u32>,
  clusters_late: atomic<u32>,  // drawn in pass 2
  // where each pass's clusters start: hardware [0, 3), software [3, 6), work
  // items [6, 8). written by the args passes, read by the rasterizers.
  pass_start: array<u32, 8>,
  big: array<atomic<u32>, 2>,  // per pass: instances for expand
  visible_cells: array<atomic<u32>, 2>,  // per pass (cell_cull)
  late_cells: atomic<u32>,               // hidden in pass 1, for pass 2
}

// pass (0 or 1), and the pyramid builder's level: a uniform bound at an offset
// per dispatch, for want of push constants.
struct PassInfo {
  pass_index: u32,
  level: u32,
  pad0: u32,
  pad1: u32,
}

@group(1) @binding(0) var<storage, read_write> counters: Counters;
@group(1) @binding(1) var<storage, read_write> work: array<vec2u>;        // (instance, first cluster)
@group(1) @binding(2) var<storage, read_write> hw_visible: array<vec2u>;  // (instance, cluster)
@group(1) @binding(3) var<storage, read_write> sw_visible: array<vec2u>;
// software rasterizer depth per pixel, then triangle per pixel: one buffer,
// storage buffers being scarce.
@group(1) @binding(4) var<storage, read_write> sw_buf: array<atomic<u32>>;
@group(2) @binding(0) var hw_depth: texture_depth_2d;
@group(2) @binding(1) var hw_id: texture_2d<u32>;
@group(2) @binding(2) var out_image: texture_storage_2d<rgba8unorm, write>;
// virtual shadow maps (vsm_common.wgsl), for shading.
@group(2) @binding(3) var<storage, read> vsm_entries: array<u32>;
@group(2) @binding(4) var<storage, read> vsm_atlas: array<u32>;
// depth pyramid: level 0 half the screen, each texel the farthest (smallest,
// reversed) depth it covers.
@group(2) @binding(5) var hzb: texture_2d<f32>;
@group(2) @binding(13) var ao_image: texture_2d<f32>;  // ao.wgsl, half resolution: indirect light, occlusion

// ambient occlusion over a full-resolution pixel.
fn occlusion(px: vec2u) -> f32 {
  if ((frame.flags & FLAG_AO) == 0u) { return 1.0; }
  return textureLoad(ao_image, min(px / 2u, textureDimensions(ao_image) - 1u), 0).a;
}

// light from all but the sun: ao.wgsl's (sky, ground bounce, nearby bounce), or the plain one.
fn indirect_light(px: vec2u, n: vec3f) -> vec3f {
  if ((frame.flags & FLAG_AO) == 0u) { return plain_ambient(n, 1.0); }
  return textureLoad(ao_image, min(px / 2u, textureDimensions(ao_image) - 1u), 0).rgb;
}
// building it: the level below, the level written.
@group(2) @binding(6) var hzb_src: texture_2d<f32>;
@group(2) @binding(7) var hzb_dst: texture_storage_2d<r32float, write>;

// screen rectangle of a view-space sphere (z forward) in [0, 1] texture
// coordinates (mara and mcguire 2013, as in zeux's niagara). false if it
// reaches the near plane.
fn project_sphere(c: vec3f, r: f32, aabb: ptr<function, vec4f>) -> bool {
  if (c.z < r + frame.near_z) { return false; }
  let cr = c * r;
  let czr2 = c.z * c.z - r * r;
  let vx = sqrt(c.x * c.x + czr2);
  let minx = (vx * c.x - cr.z) / (vx * c.z + cr.x);
  let maxx = (vx * c.x + cr.z) / (vx * c.z - cr.x);
  let vy = sqrt(c.y * c.y + czr2);
  let miny = (vy * c.y - cr.z) / (vy * c.z + cr.y);
  let maxy = (vy * c.y + cr.z) / (vy * c.z - cr.y);
  let a = vec4f(minx * frame.p00, miny * frame.p11, maxx * frame.p00, maxy * frame.p11);
  *aabb = a.xwzy * vec4f(0.5, -0.5, 0.5, -0.5) + vec4f(0.5);
  return true;
}

// whether a world-space sphere is surely hidden: nearest point farther than the
// farthest depth in its rectangle. pass 1 from last frame's camera and pyramid,
// pass 2 from this frame's.
fn occluded(center: vec3f, radius: f32) -> bool {
  if ((frame.flags & FLAG_OCCLUSION) == 0u) { return false; }
  if (pass_info.pass_index == 0u && (frame.flags & FLAG_PREV_VALID) == 0u) { return false; }
  var c = (frame.view * vec4f(center, 1.0)).xyz;
  if (pass_info.pass_index == 0u) { c = (frame.prev_view * vec4f(center, 1.0)).xyz; }
  c.z = -c.z;
  var aabb: vec4f;
  if (!project_sphere(c, radius, &aabb)) { return false; }
  let screen = vec2f(f32(frame.width), f32(frame.height));
  // a pixel's margin for the taa jitter, which the projection leaves out.
  let lo = max(clamp(aabb.xy, vec2f(0.0), vec2f(1.0)) * screen - 1.0, vec2f(0.0));
  let hi = min(clamp(aabb.zw, vec2f(0.0), vec2f(1.0)) * screen + 0.5, screen - 0.5);
  // finest level where the rectangle spans at most 2x2 texels.
  var level = 0u;
  var t0: vec2i;
  var t1: vec2i;
  loop {
    let s = exp2(f32(level + 1u));
    t0 = vec2i(floor(lo / s));
    t1 = vec2i(floor(max(hi, lo) / s));
    if (all(t1 - t0 <= vec2i(1)) || level + 1u >= frame.hzb_levels) { break; }
    level++;
  }
  let size = vec2i(textureDimensions(hzb, level)) - vec2i(1);
  t0 = min(t0, size);
  t1 = min(t1, size);
  var far_depth = textureLoad(hzb, t0, level).x;
  far_depth = min(far_depth, textureLoad(hzb, vec2i(t1.x, t0.y), level).x);
  far_depth = min(far_depth, textureLoad(hzb, vec2i(t0.x, t1.y), level).x);
  far_depth = min(far_depth, textureLoad(hzb, t1, level).x);
  return frame.near_z / (c.z - radius) < far_depth;
}

// pyramid level 0: the nearer of hardware and software depth per pixel; empty
// counts as infinitely far.
@compute @workgroup_size(8, 8)
fn hzb_first(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(hzb_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  var far_depth = 1e30;
  for (var k = 0u; k < 4u; k++) {
    let q = gid.xy * 2u + vec2u(k & 1u, k >> 1u);
    if (q.x < frame.width && q.y < frame.height) {
      let d = max(textureLoad(hw_depth, q, 0), bitcast<f32>(atomicLoad(&sw_buf[q.x + q.y * frame.width])));
      far_depth = min(far_depth, d);
    }
  }
  textureStore(hzb_dst, gid.xy, vec4f(far_depth));
}

// each later level from the one before.
@compute @workgroup_size(8, 8)
fn hzb_down(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(hzb_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let src = textureDimensions(hzb_src);
  var far_depth = 1e30;
  for (var k = 0u; k < 4u; k++) {
    let q = gid.xy * 2u + vec2u(k & 1u, k >> 1u);
    if (q.x < src.x && q.y < src.y) { far_depth = min(far_depth, textureLoad(hzb_src, q, 0).x); }
  }
  textureStore(hzb_dst, gid.xy, vec4f(far_depth));
}
// streaming: the frame each page was last drawn from, this frame's requests
// (page, priority as float bits), when each was last asked.
struct Requests {
  count: atomic<u32>,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  list: array<vec2u>,
}
// per page: frame last drawn from, then frame last asked for.
@group(1) @binding(6) var<storage, read_write> page_stamps: array<atomic<u32>>;
@group(1) @binding(9) var<uniform> pass_info: PassInfo;
// what pass 1 hid: clusters (instance, cluster) first, then instances
// (instance, 0) from max_visible.
@group(1) @binding(12) var<storage, read_write> late: array<vec2u>;
@group(1) @binding(7) var<storage, read_write> requests: Requests;

// lod test with streaming, as common.glsl's: drawn at the right level, or when
// the finer clusters it stands for are missing; it must be resident.
struct Lod {
  draw: bool,
  wants_finer: bool,
  self_error: f32,
}

fn lod_test(inst: Instance, c: Cluster) -> Lod {
  let s = inst.scale;
  var r: Lod;
  // a deforming or skinned instance: spheres where the points are, errors grown (lod_sphere(),
  // grown_error() in common.wgsl). the lod sphere grows by the creator page's joints (a leaf's,
  // its own page's), the parent sphere by its own page's: one computation from either side.
  let lod_page = select(c.group, c.creator, c.creator != NO_PAGE);
  let lod = lod_sphere(inst, lod_page, c.lod_center, c.lod_radius);
  let parent = lod_sphere(inst, c.group, c.parent_center, c.parent_radius);
  let lod_e = grown_error(inst, c.lod_error) * s;
  let parent_e = grown_error(inst, c.parent_error) * s;
  r.self_error = projected_error(to_world(inst, lod.xyz), lod.w * s, lod_e);
  var parent_error = projected_error(to_world(inst, parent.xyz), parent.w * s, parent_e);
  // skinned: either nested sphere bounds it, and the smaller of two nested bounds nests too.
  if (skinned(inst)) {
    let lod_r = rest_sphere(inst, lod_page, c.lod_center, c.lod_radius);
    let parent_r = rest_sphere(inst, c.group, c.parent_center, c.parent_radius);
    r.self_error = min(r.self_error, projected_error(to_world(inst, lod_r.xyz), lod_r.w * s, lod_e));
    parent_error = min(parent_error, projected_error(to_world(inst, parent_r.xyz), parent_r.w * s, parent_e));
  }
  let coarse_enough = parent_error > frame.lod_threshold;
  let finer_resident = c.creator != NO_PAGE && page_table[c.creator] != NO_PAGE;
  r.wants_finer = r.self_error > frame.lod_threshold && c.creator != NO_PAGE && !finer_resident;
  r.draw = page_table[c.group] != NO_PAGE && coarse_enough && (r.self_error <= frame.lod_threshold || !finer_resident);
  return r;
}

fn request_finer(c: Cluster, priority: f32) {
  let stamp = arrayLength(&page_stamps) / 2u + c.creator;
  if (atomicExchange(&page_stamps[stamp], frame.frame_index) == frame.frame_index) { return; }
  let k = atomicAdd(&requests.count, 1u);
  if (k < frame.max_requests) { requests.list[k] = vec2u(c.creator, bitcast<u32>(priority)); }
}

// a model's texture, streamed as tiles into the page pool (include/texture_file.hpp, and
// viewer/shaders/texture.glsl, which this follows): bc1 decoded here, bilinear within a tile
// (its border holds the neighbours), trilinear across levels. a missing tile is asked for and
// its nearest resident coarser copy drawn: tiles are resident only while their coarser copies
// are, and the coarsest is pinned.
const TILE_TEXELS = 128u;
const TILE_BORDER = 4u;
const TILE_PAYLOAD = 120u;

fn srgb_to_linear3(c: vec3f) -> vec3f {
  return select(c / 12.92, pow((c + 0.055) / 1.055, vec3f(2.4)), c > vec3f(0.04045));
}

fn unpack565(v: u32) -> vec3f {
  return vec3f(f32(v >> 11u), f32((v >> 5u) & 63u), f32(v & 31u)) / vec3f(31.0, 63.0, 31.0);
}

// texel t (0 to 127 each way) of the tile at word `at`, linear.
fn tile_texel(at: u32, t: vec2u) -> vec3f {
  let block = at + 2u * ((t.y >> 2u) * (TILE_TEXELS / 4u) + (t.x >> 2u));
  let ends = pool[block];
  let bits = pool[block + 1u];
  let c0 = ends & 0xffffu;
  let c1 = ends >> 16u;
  let index = (bits >> (2u * ((t.y & 3u) * 4u + (t.x & 3u)))) & 3u;
  let a = unpack565(c0);
  let b = unpack565(c1);
  var c = a;
  if (index == 1u) { c = b; }
  else if (index == 2u) { c = select((a + b) * 0.5, (2.0 * a + b) / 3.0, c0 > c1); }
  else if (index == 3u) { c = select(vec3f(0.0), (a + 2.0 * b) / 3.0, c0 > c1); }
  return srgb_to_linear3(c);
}

struct TextureLevel {
  size: vec2u,
  tiles: vec2u,
  first: u32,
}

fn texture_level(tex: vec4u, level: u32) -> TextureLevel {
  var l: TextureLevel;
  l.size = tex.zw;
  l.first = tex.x;
  for (var k = 0u; k < level; k++) {
    l.tiles = (l.size + TILE_PAYLOAD - 1u) / TILE_PAYLOAD;
    l.first += l.tiles.x * l.tiles.y;
    l.size = (l.size + 1u) / 2u;
  }
  l.tiles = (l.size + TILE_PAYLOAD - 1u) / TILE_PAYLOAD;
  return l;
}

// bilinear at uv from the finest resident level from `level` up; a missing `level` tile is
// asked for, more urgently the coarser the copy drawn instead.
fn sample_resident(tex: vec4u, uv: vec2f, level: u32) -> vec3f {
  var wanted = NO_PAGE;
  for (var k = level; k < tex.y; k++) {
    let l = texture_level(tex, k);
    let at = clamp(uv, vec2f(0.0), vec2f(1.0)) * vec2f(l.size) - 0.5;
    let base = floor(at);
    let f = at - base;
    let b = vec2i(base);
    let tile = min(vec2u(max(b, vec2i(0))) / TILE_PAYLOAD, l.tiles - 1u);
    let page = l.first + tile.y * l.tiles.x + tile.x;
    let word = page_table[page];
    if (word == NO_PAGE) {
      if (k == level) { wanted = page; }
      continue;
    }
    if (wanted != NO_PAGE) {
      let stamp = arrayLength(&page_stamps) / 2u + wanted;
      if (atomicExchange(&page_stamps[stamp], frame.frame_index) != frame.frame_index) {
        let n = atomicAdd(&requests.count, 1u);
        if (n < frame.max_requests) { requests.list[n] = vec2u(wanted, bitcast<u32>(frame.lod_threshold * exp2(f32(k - level)))); }
      }
    }
    atomicStore(&page_stamps[page], frame.frame_index);
    let t = vec2u(clamp(b - vec2i(tile * TILE_PAYLOAD) + i32(TILE_BORDER), vec2i(0), vec2i(i32(TILE_TEXELS) - 2)));
    return mix(mix(tile_texel(word, t), tile_texel(word, t + vec2u(1u, 0u)), f.x),
               mix(tile_texel(word, t + vec2u(0u, 1u)), tile_texel(word, t + vec2u(1u, 1u)), f.x), f.y);
  }
  return vec3f(1.0, 0.0, 1.0);  // unreachable: the coarsest tile is pinned
}

// trilinear: the level whose texels match the pixel's footprint, and the next.
fn sample_texture(tex: vec4u, uv: vec2f, duv_dx: vec2f, duv_dy: vec2f) -> vec3f {
  let size = vec2f(tex.zw);
  let footprint = max(length(duv_dx * size), length(duv_dy * size));
  let lod = clamp(log2(max(footprint, 1e-8)), 0.0, f32(tex.y - 1u));
  let level = u32(lod);
  let f = lod - f32(level);
  let a = sample_resident(tex, uv, level);
  if (f < 1.0 / 64.0 || level + 1u >= tex.y) { return a; }
  return mix(a, sample_resident(tex, uv, level + 1u), f);
}

// weights of a triangle's corners 1 and 2 where the ray from the camera along dir meets its
// plane, unclamped.
fn plane_bary(p0: vec3f, p1: vec3f, p2: vec3f, origin: vec3f, dir: vec3f) -> vec2f {
  let e1 = p1 - p0;
  let e2 = p2 - p0;
  let pv = cross(dir, e2);
  let det = dot(e1, pv);
  let inv = select(0.0, 1.0 / det, abs(det) > 1e-20);
  let tv = origin - p0;
  return vec2f(dot(tv, pv), dot(dir, cross(tv, e1))) * inv;
}

// indirect arguments: [0, 8) cluster culling per pass, [8, 16) hardware draws,
// [16, 24) software dispatches, [24, 28) pass 2 instance culling, [28, 36)
// expand per pass, [36, 40) pass 1 instance culling, [40, 44) pass 2 cell
// culling. bound only for the passes writing them: a dispatch may not read its
// arguments from a buffer bound for writing.

// cells of up to 64 neighbouring instances, culled before their instances are
// read (as cell_cull.comp). bound only for cell and instance culling, in place
// of the image group, so 16 storage buffers still suffice.
struct Cell {
  center: vec3f,
  radius: f32,
  first: u32,
  count: u32,
  pad0: u32,
  pad1: u32,
}
@group(2) @binding(14) var<storage, read> cells: array<Cell>;
// cells pass 1 hid [0, n), then the visible cells of each pass from n and 2n
// (n cells).
@group(2) @binding(15) var<storage, read_write> cell_list: array<u32>;
@group(3) @binding(0) var<storage, read_write> args: array<u32>;


// an invocation per cell: pass 1 tests frustum and last frame's depth and lists
// hidden cells; pass 2 tests those against pass 1's depth. visible cells go to
// instance_cull.
@compute @workgroup_size(64)
fn cell_cull(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  let pass_index = pass_info.pass_index;
  let total = arrayLength(&cells);
  let n = select(atomicLoad(&counters.late_cells), total, pass_index == 0u);
  let k = (wid.y * 65535u + wid.x) * 64u + lane;
  if (k >= n) { return; }
  let c = select(cell_list[k], k, pass_index == 0u);
  let cell = cells[c];
  if (pass_index == 0u && (frame.flags & FLAG_FRUSTUM) != 0u && !sphere_in_frustum(cell.center, cell.radius)) { return; }
  // the sphere holds moving instances wherever they go, last frame included.
  if (occluded(cell.center, cell.radius)) {
    if (pass_index == 0u) { cell_list[atomicAdd(&counters.late_cells, 1u)] = c; }
    return;
  }
  cell_list[total * (1u + pass_index) + atomicAdd(&counters.visible_cells[pass_index], 1u)] = c;
}

// instance culling's dispatch: a workgroup per visible cell, and in pass 2 first
// one per 64 instances pass 1 hid. pass 1 also sizes pass 2's cell culling.
@compute @workgroup_size(1)
fn args_cells() {
  let pass_index = pass_info.pass_index;
  var groups = atomicLoad(&counters.visible_cells[pass_index]);
  if (pass_index == 1u) { groups += (atomicLoad(&counters.late_instances) + 63u) / 64u; }
  rows(groups, select(36u, 24u, pass_index == 1u));
  if (pass_index == 0u) { rows((atomicLoad(&counters.late_cells) + 63u) / 64u, 40u); }
}

// a workgroup per visible cell, an invocation per instance (as
// instance_cull.comp): the workgroup writes its instances' work items after a
// prefix sum of their counts. instances over BIG_PIECES (near ones, neighbours
// in the list) go to expand, a workgroup each.
//
// pass 1 tests against last frame's depth and lists what it hides; pass 2 runs
// that list, then the cells pass 2 found visible again, against pass 1's depth.
const BIG_PIECES = 16u;

var<workgroup> wg_ends: array<u32, 64>;
var<workgroup> wg_instance: array<u32, 64>;
var<workgroup> wg_first: array<u32, 64>;
var<workgroup> wg_base: u32;

// each pass's big instances in late: two entries each, (instance, first
// cluster) and (pieces, 0).
fn big_slot(pass_index: u32, k: u32) -> u32 {
  return frame.max_visible + frame.instance_count * (1u + 2u * pass_index) + 2u * k;
}

@compute @workgroup_size(64)
fn instance_cull(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  let pass_index = pass_info.pass_index;
  let group = wid.y * 65535u + wid.x;
  // which instance: a visible cell's, or in pass 2 first those pass 1 hid (inside the
  // frustum already).
  let late_count = atomicLoad(&counters.late_instances);
  let late_groups = select(0u, (late_count + 63u) / 64u, pass_index == 1u);
  var valid = false;
  var hidden_before = false;
  var i = 0u;
  if (group < late_groups) {
    let slot = group * 64u + lane;
    valid = slot < late_count;
    if (valid) { i = late[frame.max_visible + slot].x; }
    hidden_before = true;
  } else {
    let cell = cells[cell_list[arrayLength(&cells) * (1u + pass_index) + group - late_groups]];
    valid = lane < cell.count;
    i = cell.first + lane;
  }
  var chunks = 0u;
  var first = 0u;
  if (valid) {
    let inst = load_instance(i);
    let m = meshes[inst.mesh];
    let sphere = instance_sphere(inst, m.bounds, false);
    let center = to_world(inst, sphere.xyz);
    let radius = sphere.w * inst.scale;
    var keep = hidden_before || (frame.flags & FLAG_FRUSTUM) == 0u || sphere_in_frustum(center, radius);
    // pass 1: was it hidden last frame, where it was last frame (skinned: where it is now,
    // grown by how far a point moved since).
    var then = center;
    var then_r = radius;
    if (pass_index == 0u && inst.anim != 0u && !skinned(inst)) { then = to_world(load_prev_instance(i), m.bounds.xyz); }
    if (pass_index == 0u && skinned(inst)) { then_r += pose_info(inst).z * (frame.time - frame.prev_time) * inst.scale; }
    if (keep && occluded(then, then_r)) {
      if (pass_index == 0u) { late[frame.max_visible + atomicAdd(&counters.late_instances, 1u)] = vec2u(i, 0u); }
      else { atomicAdd(&counters.instances_occluded, 1u); }
      keep = false;
    }
    if (keep) {
      atomicAdd(&counters.instances_visible, 1u);
      // projected_error()'s worst over the sphere holding every lod sphere.
      let ls = instance_sphere(inst, m.lod_bounds, true);
      let lc = to_world(inst, ls.xyz) - frame.cull_origin.xyz;
      let r = ls.w * inst.scale;
      let z = max(dot(lc, frame.cull_planes[4].xyz) - r, frame.near_z);
      let limit = error_floor(inst, frame.lod_threshold * z / (inst.scale * frame.lod_scale * projected_sec(lc, r)));
      var lo = 0u;
      var hi = m.cluster_count;
      while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (cluster_parent_error(m.first_cluster + mid) > limit) { hi = mid; } else { lo = mid + 1u; }
      }
      chunks = (m.cluster_count - lo + 63u) / 64u;
      first = m.first_cluster + lo;
      if (chunks > BIG_PIECES) {
        let at = big_slot(pass_index, atomicAdd(&counters.big[pass_index], 1u));
        late[at] = vec2u(i, first);
        late[at + 1u] = vec2u(chunks, 0u);
        chunks = 0u;
      }
    }
  }

  // prefix sum over the workgroup (hillis and steele).
  wg_ends[lane] = chunks;
  wg_instance[lane] = i;
  wg_first[lane] = first;
  workgroupBarrier();
  for (var step = 1u; step < 64u; step <<= 1u) {
    var add = 0u;
    if (lane >= step) { add = wg_ends[lane - step]; }
    workgroupBarrier();
    wg_ends[lane] += add;
    workgroupBarrier();
  }
  var total = workgroupUniformLoad(&wg_ends[63]);
  if (total == 0u) { return; }
  if (lane == 0u) { wg_base = atomicAdd(&counters.work[pass_index], total); }
  let at = workgroupUniformLoad(&wg_base);
  if (at + total > frame.max_work) {
    if (lane == 0u) { atomicAdd(&counters.overflow, 1u); }
    total = select(0u, frame.max_work - at, at < frame.max_work);
  }
  for (var k = lane; k < total; k += 64u) {
    // invocation whose run holds item k: the first end past it.
    var lo = 0u;
    var hi = 63u;
    while (lo < hi) {
      let mid = (lo + hi) / 2u;
      if (wg_ends[mid] > k) { hi = mid; } else { lo = mid + 1u; }
    }
    var start = 0u;
    if (lo > 0u) { start = wg_ends[lo - 1u]; }
    work[pass_index * frame.max_work + at + k] = vec2u(wg_instance[lo], wg_first[lo] + 64u * (k - start));
  }
}

// work items of instances with many, a workgroup each.
var<workgroup> wg_expand_base: u32;
var<workgroup> wg_expand_count: u32;

@compute @workgroup_size(64)
fn expand(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  let pass_index = pass_info.pass_index;
  let k = wid.y * 65535u + wid.x;
  if (lane == 0u) { wg_expand_count = atomicLoad(&counters.big[pass_index]); }
  if (k >= workgroupUniformLoad(&wg_expand_count)) { return; }
  let at = big_slot(pass_index, k);
  let e = late[at];
  if (lane == 0u) {
    var chunks = late[at + 1u].x;
    let b = atomicAdd(&counters.work[pass_index], chunks);
    if (b + chunks > frame.max_work) {
      atomicAdd(&counters.overflow, 1u);
      chunks = select(0u, frame.max_work - b, b < frame.max_work);
    }
    wg_expand_base = pass_index * frame.max_work + b;
    wg_expand_count = chunks;
  }
  let base = workgroupUniformLoad(&wg_expand_base);
  let count = workgroupUniformLoad(&wg_expand_count);
  for (var c = lane; c < count; c += 64u) {
    work[base + c] = vec2u(e.x, e.y + 64u * c);
  }
}

fn rows(n: u32, at: u32) {
  args[at] = min(n, 65535u);
  args[at + 1u] = (n + 65534u) / 65535u;
  args[at + 2u] = 1u;
}

// after instance culling: the expand dispatch.
@compute @workgroup_size(1)
fn args_big() {
  let pass_index = pass_info.pass_index;
  rows(atomicLoad(&counters.big[pass_index]), 28u + pass_index * 4u);
}

// cluster culling dispatch: a workgroup per work item, plus in pass 2 one per
// 64 hidden clusters.
@compute @workgroup_size(1)
fn args_cull() {
  let pass_index = pass_info.pass_index;
  let items = min(atomicLoad(&counters.work[pass_index]), frame.max_work);
  var n = items;
  if (pass_index == 1u) { n += (min(atomicLoad(&counters.late_clusters), frame.max_visible) + 63u) / 64u; }
  rows(n, pass_index * 4u);
  counters.pass_start[6u + pass_index] = items;
  if (pass_index == 0u) {
    counters.pass_start[0] = 0u;
    counters.pass_start[3] = 0u;
  }
}

// --- cluster culling: a workgroup per work item, an invocation per cluster:
// lod cut, frustum, normal cone. small survivors go to the software rasterizer,
// the rest to the hardware.

var<workgroup> wg_hw: atomic<u32>;
var<workgroup> wg_sw: atomic<u32>;
var<workgroup> wg_tested: atomic<u32>;
var<workgroup> wg_triangles: atomic<u32>;
var<workgroup> wg_hw_base: u32;
var<workgroup> wg_sw_base: u32;

@compute @workgroup_size(64)
fn cluster_cull(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  if (lane == 0u) {
    atomicStore(&wg_hw, 0u);
    atomicStore(&wg_sw, 0u);
    atomicStore(&wg_tested, 0u);
    atomicStore(&wg_triangles, 0u);
  }
  workgroupBarrier();
  let pass_index = pass_info.pass_index;
  let item = wid.y * 65535u + wid.x;
  var draw = false;
  var software = false;
  var hidden = false;
  var instance_id = 0u;
  var cluster_id = 0u;
  let items = counters.pass_start[6u + pass_index];
  if (item < items) {
    let w = work[pass_index * frame.max_work + item];
    let inst = load_instance(w.x);
    let m = meshes[inst.mesh];
    instance_id = w.x;
    cluster_id = w.y + lane;
    if (cluster_id < m.first_cluster + m.cluster_count) {
      atomicAdd(&wg_tested, 1u);
      let c = load_cluster(cluster_id);
      let s = inst.scale;
      let lod = lod_test(inst, c);
      draw = lod.draw;
      let drawn = drawn_sphere(inst, c.group, c.center, c.radius);
      let center = to_world(inst, drawn.xyz);
      let r = drawn.w * s;
      if (draw && (frame.flags & FLAG_FRUSTUM) != 0u) { draw = sphere_in_frustum(center, r); }
      // (a deforming or skinned instance's normals turn: no cone.)
      if (draw && (frame.flags & FLAG_CONE) != 0u && c.cone_cutoff < 1.0 && !deforming(inst) && !skinned(inst)) {
        let axis = normalize(to_world_dir(inst, c.cone_axis));
        let view = center - frame.cull_origin.xyz;
        if (dot(view, axis) >= c.cone_cutoff * length(view) + r) { draw = false; }
      }
      var then = center;
      var then_r = r;
      if (pass_index == 0u && inst.anim != 0u && !skinned(inst)) { then = to_world(load_prev_instance(w.x), c.center); }
      if (pass_index == 0u && skinned(inst)) { then_r += pose_info(inst).z * (frame.time - frame.prev_time) * s; }
      if (draw && occluded(then, then_r)) {
        draw = false;
        hidden = true;
      }
      if (draw) {
        atomicAdd(&wg_triangles, c.triangle_count);
        atomicStore(&page_stamps[c.group], frame.frame_index);
        if (lod.wants_finer) { request_finer(c, lod.self_error); }
        // rough size on screen; clusters at the near plane go to the hardware,
        // which clips.
        let d = length(center - frame.origin.xyz) - r;
        software = (frame.flags & FLAG_SOFTWARE) != 0u && d > frame.near_z * 2.0 &&
                   2.0 * r * frame.lod_scale / d < frame.sw_max_pixels;
      }
    }
  } else if (pass_index == 1u) {
    // pass 1's hidden clusters: only occlusion left to test.
    let k = (item - items) * 64u + lane;
    if (k < min(atomicLoad(&counters.late_clusters), frame.max_visible)) {
      atomicAdd(&wg_tested, 1u);
      let ic = late[k];
      instance_id = ic.x;
      cluster_id = ic.y;
      let inst = load_instance(instance_id);
      let c = load_cluster(cluster_id);
      let drawn = drawn_sphere(inst, c.group, c.center, c.radius);
      let center = to_world(inst, drawn.xyz);
      let r = drawn.w * inst.scale;
      draw = !occluded(center, r);
      if (draw) {
        atomicAdd(&wg_triangles, c.triangle_count);
        atomicAdd(&counters.clusters_late, 1u);
        atomicStore(&page_stamps[c.group], frame.frame_index);
        let lod = lod_test(inst, c);
        if (lod.wants_finer) { request_finer(c, lod.self_error); }
        let d = length(center - frame.origin.xyz) - r;
        software = (frame.flags & FLAG_SOFTWARE) != 0u && d > frame.near_z * 2.0 &&
                   2.0 * r * frame.lod_scale / d < frame.sw_max_pixels;
      }
    }
  }
  if (hidden && pass_index == 0u) {
    let k = atomicAdd(&counters.late_clusters, 1u);
    if (k < frame.max_visible) { late[k] = vec2u(instance_id, cluster_id); }
  }
  var slot = 0u;
  if (draw) {
    if (software) { slot = atomicAdd(&wg_sw, 1u); } else { slot = atomicAdd(&wg_hw, 1u); }
  }
  workgroupBarrier();
  if (lane == 0u) {
    let h = atomicLoad(&wg_hw);
    let s = atomicLoad(&wg_sw);
    if (h > 0u) { wg_hw_base = atomicAdd(&counters.hw, h); }
    if (s > 0u) { wg_sw_base = atomicAdd(&counters.sw, s); }
    atomicAdd(&counters.clusters_tested, atomicLoad(&wg_tested));
    atomicAdd(&counters.triangles, atomicLoad(&wg_triangles));
  }
  workgroupBarrier();
  if (draw) {
    if (software) {
      let k = wg_sw_base + slot;
      if (k < frame.max_visible) { sw_visible[k] = vec2u(instance_id, cluster_id); }
    } else {
      let k = wg_hw_base + slot;
      if (k < frame.max_visible) { hw_visible[k] = vec2u(instance_id, cluster_id); }
    }
  }
}

// a pass's draws: hardware (384 vertices per cluster, an instance per cluster)
// and software, over the clusters this pass added.
@compute @workgroup_size(1)
fn args_draw() {
  let pass_index = pass_info.pass_index;
  let hw_end = min(atomicLoad(&counters.hw), frame.max_visible);
  let sw_end = min(atomicLoad(&counters.sw), frame.max_visible);
  counters.pass_start[pass_index + 1u] = hw_end;
  counters.pass_start[4u + pass_index] = sw_end;
  let at = 8u + pass_index * 4u;
  args[at] = 384u;
  args[at + 1u] = hw_end - counters.pass_start[pass_index];
  args[at + 2u] = 0u;
  args[at + 3u] = 0u;
  rows(sw_end - counters.pass_start[3u + pass_index], 16u + pass_index * 4u);
}

// --- software rasterizer: a workgroup per small cluster, an invocation per
// vertex then per triangle. 1/256 pixel snapping, edges from each triangle's
// corner to fit 32 bits. pass 1 keeps the nearest depth; pass 2 writes the
// triangle where its depth won. triangles grow by a step (1/256): where this
// meets the render pipeline, a vertex can land a step apart and leave a sliver
// neither draws (viewer/shaders/sw_raster.comp).

var<workgroup> screen: array<vec2i, 128>;
var<workgroup> depth: array<f32, 128>;
var<workgroup> wg_valid: u32;

fn edge(a: vec2i, b: vec2i, p: vec2i) -> i32 {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn sw_raster(wid: vec3u, lane: u32, write_id: bool) {
  let k = counters.pass_start[3u + pass_info.pass_index] + wid.y * 65535u + wid.x;
  if (lane == 0u) { wg_valid = select(0u, 1u, k < counters.pass_start[4u + pass_info.pass_index]); }
  if (workgroupUniformLoad(&wg_valid) == 0u) { return; }
  let v = sw_visible[k];
  let inst = load_instance(v.x);
  let c = load_cluster(v.y);
  if (lane < c.vertex_count) {
    let m = meshes[inst.mesh];
    let clip = frame.view_proj * vec4f(to_world(inst, drawn_vertex(inst, c, lane)), 1.0);
    let ndc = clip.xy / clip.w;
    // pixels run down; clip space up.
    let px = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * vec2f(f32(frame.width), f32(frame.height));
    screen[lane] = vec2i(round(px * 256.0));
    depth[lane] = clip.z / clip.w;
  }
  workgroupBarrier();
  if (lane >= c.triangle_count) { return; }

  let packed = cluster_triangle(c, lane);
  let i0 = packed & 255u;
  var i1 = (packed >> 8u) & 255u;
  var i2 = (packed >> 16u) & 255u;
  var a = screen[i0];
  var b = screen[i1];
  var d = screen[i2];
  // the box widened as the triangle grows.
  let lo_abs = max((min(a, min(b, d)) - vec2i(2 + 128) + vec2i(255)) >> vec2u(8u), vec2i(0));
  let hi = min((max(a, max(b, d)) + vec2i(2 - 128)) >> vec2u(8u), vec2i(i32(frame.width), i32(frame.height)) - vec2i(1)) - lo_abs;
  if (hi.x < 0 || hi.y < 0) { return; }
  let origin = lo_abs * 256;
  a -= origin;
  b -= origin;
  d -= origin;
  var area = edge(a, b, d);
  if (area == 0) { return; }
  var za = depth[i0];
  var zb = depth[i1];
  var zd = depth[i2];
  // both sides drawn: one winding.
  if (area < 0) {
    let t = b; b = d; d = t;
    let tz = zb; zb = zd; zd = tz;
    area = -area;
  }
  // each edge out by a step: an edge function is its length times the distance, and
  // |x| + |y| is at least the length.
  let bias0 = abs(d.x - b.x) + abs(d.y - b.y);
  let bias1 = abs(a.x - d.x) + abs(a.y - d.y);
  let bias2 = abs(b.x - a.x) + abs(b.y - a.y);
  let p0 = vec2i(128);
  var row0 = edge(b, d, p0) + bias0;
  var row1 = edge(d, a, p0) + bias1;
  var row2 = edge(a, b, p0) + bias2;
  let dx0 = (d.y - b.y) * -256; let dy0 = (d.x - b.x) * 256;
  let dx1 = (a.y - d.y) * -256; let dy1 = (a.x - d.x) * 256;
  let dx2 = (b.y - a.y) * -256; let dy2 = (b.x - a.x) * 256;
  let inv_area = 1.0 / f32(area);
  let id = (k << 7u) | lane;
  for (var y = 0; y <= hi.y; y++) {
    var w0 = row0; var w1 = row1; var w2 = row2;
    let row = u32(lo_abs.y + y) * frame.width + u32(lo_abs.x);
    for (var x = 0; x <= hi.x; x++) {
      if ((w0 | w1 | w2) >= 0) {
        let z = (f32(w0 - bias0) * za + f32(w1 - bias1) * zb + f32(w2 - bias2) * zd) * inv_area;
        let bits = bitcast<u32>(z);
        let at = row + u32(x);
        if (write_id) {
          if (atomicLoad(&sw_buf[at]) == bits) { atomicStore(&sw_buf[frame.width * frame.height + at], id); }
        } else if (bits > atomicLoad(&sw_buf[at])) {
          atomicMax(&sw_buf[at], bits);
        }
      }
      w0 += dx0; w1 += dx1; w2 += dx2;
    }
    row0 += dy0; row1 += dy1; row2 += dy2;
  }
}

@compute @workgroup_size(128)
fn sw_depth_pass(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  sw_raster(wid, lane, false);
}

@compute @workgroup_size(128)
fn sw_id_pass(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  sw_raster(wid, lane, true);
}

// --- shading: the nearer of the hardware and software results, its triangle
// refetched and hit with the pixel's ray.

fn hash_color(x: u32) -> vec3f {
  let h = hash(x);
  return vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0 * 0.8 + 0.1;
}

fn level_color(level: u32) -> vec3f {
  let h = 0.66 - 0.055 * f32(min(level, 16u));
  let k = abs(fract(vec3f(h) + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return mix(vec3f(1.0), clamp(k - 1.0, vec3f(0.0), vec3f(1.0)), 0.75) * 0.85;
}

// per instance (Instance::material), as shade.comp's.
struct Material {
  albedo: vec3f,
  roughness: f32,
  metallic: f32,
}

fn material(index: u32) -> Material {
  switch (index) {
    case 1u: { return Material(vec3f(0.66, 0.64, 0.6), 0.3, 0.0); }    // polished marble
    case 2u: { return Material(vec3f(0.62, 0.5, 0.38), 0.85, 0.0); }   // sandstone
    case 3u: { return Material(vec3f(0.58, 0.38, 0.22), 0.35, 1.0); }  // bronze
    case 4u: { return Material(vec3f(1.0, 0.7, 0.27), 0.28, 1.0); }    // gold
    case 5u: { return Material(vec3f(0.2, 0.2, 0.22), 0.45, 0.0); }    // dark granite
    default: { return Material(vec3f(0.56, 0.52, 0.47), 0.6, 0.0); }   // plaster
  }
}

// procedural surface detail, as shade.comp's: 3d value noise with its gradient in the
// model's own space over its radius, octaves finer than a pixel faded out; the gradient
// tilts the normal.
fn value_noise(p: vec3f) -> vec4f {
  let i = vec3i(floor(p));
  let f = p - floor(p);
  let u = f * f * (3.0 - 2.0 * f);
  let du = 6.0 * f * (1.0 - f);
  var v: array<f32, 8>;
  for (var k = 0; k < 8; k++) {
    let c = i + vec3i(k & 1, (k >> 1u) & 1, k >> 2u);
    v[k] = f32(hash((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u)) & 0xffffu) / 65535.0;
  }
  let k1 = v[1] - v[0];
  let k2 = v[2] - v[0];
  let k3 = v[4] - v[0];
  let k4 = v[0] - v[1] - v[2] + v[3];
  let k5 = v[0] - v[2] - v[4] + v[6];
  let k6 = v[0] - v[1] - v[4] + v[5];
  let k7 = -v[0] + v[1] + v[2] - v[3] + v[4] - v[5] - v[6] + v[7];
  let value = v[0] + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
  let grad = du * vec3f(k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z, k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
                        k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);
  return vec4f(value, grad);
}

fn fbm_d(p: vec3f, freq0: f32, pixel: f32) -> vec4f {
  var sum = vec4f(0.0);
  var amp = 0.5;
  var total = 0.0;
  var freq = freq0;
  for (var o = 0; o < 4; o++) {
    let fade = clamp(2.0 - 4.0 * freq * pixel, 0.0, 1.0);
    let n = value_noise(p * freq + f32(o) * 17.3);
    sum += amp * fade * vec4f(n.x, n.yzw * freq);
    total += amp * fade;
    freq *= 2.0;
    amp *= 0.5;
  }
  return select(vec4f(0.5, 0.0, 0.0, 0.0), sum / total, total > 0.0);
}

fn fbm(p: vec3f, freq: f32, pixel: f32) -> f32 { return fbm_d(p, freq, pixel).x; }

fn vein(phase: f32, sharp: f32, width: f32) -> f32 {
  let line = pow(1.0 - abs(sin(phase)), sharp);
  return mix(line, 1.0 / sharp, clamp(width * sharp * 0.5 - 0.5, 0.0, 1.0));
}

struct Detail {
  m: Material,
  grad: vec3f,  // height gradient, model space over its radius
  wrap: f32,    // light past the terminator
  coat: f32,    // polish lobe
}

fn surface_detail(m: Material, kind: u32, q: vec3f, pixel: f32, ao: f32, up: f32) -> Detail {
  var d = Detail(m, vec3f(0.0), 0.0, 0.0);
  let recess = clamp((0.88 - ao) / 0.45, 0.0, 1.0);
  if (kind == 1u) {  // marble
    let warp = fbm_d(q, 1.5, pixel);
    let fine = fbm_d(q + 3.1, 5.0, pixel);
    let axis = vec3f(12.0, 20.0, 7.0);
    let phase = dot(q, axis) + 14.0 * warp.x + 4.0 * fine.x;
    let width = length(axis + 14.0 * warp.yzw + 4.0 * fine.yzw) * pixel;
    let veins = clamp(0.9 * vein(phase, 7.0, width) + 0.5 * vein(phase * 2.3 + 6.0 * fine.x, 16.0, width * 2.3), 0.0, 1.0);
    let cloud = fbm(q + 9.0, 3.0, pixel);
    d.m.albedo = m.albedo * mix(vec3f(0.97, 0.97, 1.0), vec3f(1.03, 1.0, 0.95), cloud);
    d.m.albedo = mix(d.m.albedo, vec3f(0.3, 0.32, 0.37), veins);
    d.m.albedo *= mix(vec3f(1.0), vec3f(0.86, 0.82, 0.76), recess);
    d.m.roughness = 0.22 + 0.25 * recess;
    d.grad = fine.yzw * 0.0015;
    d.wrap = 0.45;
    d.coat = 0.3 * (1.0 - recess);
  } else if (kind == 2u) {  // sandstone
    let warp = fbm_d(q, 1.5, pixel);
    let strata = q.y * 9.0 + 0.7 * warp.x;
    let band = fract(strata);
    let layer = hash(u32(i32(floor(strata)) + 4096)) % 3u;
    let next = hash(u32(i32(floor(strata)) + 4097)) % 3u;
    var colours = array<vec3f, 3>(vec3f(0.68, 0.53, 0.38), vec3f(0.78, 0.68, 0.52), vec3f(0.6, 0.41, 0.29));
    var albedo = mix(colours[layer], colours[next], smoothstep(0.8, 1.0, band));
    let afar = clamp(pixel * 9.0, 0.0, 1.0);
    albedo = mix(albedo, vec3f(0.7, 0.6, 0.47), afar);
    let grain = fbm_d(q, 70.0, pixel);
    let pits = fbm_d(q + 7.0, 22.0, pixel);
    let pit = smoothstep(0.55, 0.75, pits.x);
    d.m.albedo = albedo * (0.9 + 0.25 * (grain.x - 0.5)) * (1.0 - 0.25 * pit) * mix(1.0, 0.72, recess);
    d.m.roughness = 0.92;
    let ledge = smoothstep(0.0, 0.12, band) * (1.0 - smoothstep(0.85, 1.0, band));
    d.grad = grain.yzw * 0.0025 + pits.yzw * 0.006 * pit + vec3f(0.0, 9.0, 0.0) * 0.004 * (ledge - 0.5) * (1.0 - afar);
    d.wrap = 0.15;
  } else if (kind == 3u) {  // bronze
    let base = fbm_d(q, 3.0, pixel);
    let crust = fbm_d(q + 11.0, 30.0, pixel);
    let streaks = fbm(vec3f(q.x * 25.0, q.y * 1.2, q.z * 25.0), 1.0, pixel * 25.0);
    var patina = smoothstep(0.35, 0.75, base.x * 0.6 + recess * 0.7 + max(up, 0.0) * 0.25);
    patina = clamp(patina + 0.35 * smoothstep(0.55, 0.8, streaks) * (1.0 - abs(up)), 0.0, 1.0);
    let wear = smoothstep(0.93, 1.0, ao) * smoothstep(0.55, 0.7, fbm(q, 8.0, pixel));
    patina *= 1.0 - wear;
    let metal = mix(mix(m.albedo, vec3f(0.3, 0.2, 0.12), 0.45 * base.x), vec3f(0.78, 0.53, 0.32), wear);
    let green = mix(vec3f(0.2, 0.44, 0.38), vec3f(0.32, 0.5, 0.3), crust.x);
    d.m.albedo = mix(metal, green, patina);
    d.m.metallic = 1.0 - patina;
    d.m.roughness = mix(mix(0.35, 0.18, wear), 0.8, patina);
    d.grad = crust.yzw * 0.004 * patina;
  } else if (kind == 4u) {  // gold
    let hammer = fbm_d(q, 18.0, pixel);
    d.m.albedo = mix(m.albedo, vec3f(0.25, 0.18, 0.1), recess * 0.8);
    d.m.metallic = 1.0 - recess * 0.8;
    d.m.roughness = mix(0.16 + 0.15 * (fbm(q, 40.0, pixel) - 0.5), 0.8, recess);
    d.grad = hammer.yzw * 0.006;
  } else if (kind == 5u) {  // dark granite
    let a = value_noise(q * 140.0);
    let b = value_noise(q * 230.0 + 5.0);
    let mica = smoothstep(0.74, 0.8, b.x);
    let feldspar = smoothstep(0.58, 0.64, a.x) * (1.0 - mica);
    let quartz = vec3f(0.26, 0.26, 0.28);
    let pink = vec3f(0.46, 0.36, 0.33);
    let black = vec3f(0.03, 0.03, 0.035);
    var albedo = mix(mix(quartz, pink, feldspar), black, mica);
    albedo = mix(albedo, quartz * 0.62 + pink * 0.3 + black * 0.08, clamp(230.0 * pixel * 2.0 - 0.5, 0.0, 1.0));
    d.m.albedo = albedo * mix(1.0, 0.8, recess);
    d.m.roughness = mix(0.15, 0.4, max(feldspar, mica));
    d.grad = b.yzw * 230.0 * 0.00015 * clamp(1.5 - 230.0 * pixel * 2.0, 0.0, 1.0);
    d.coat = 0.4;
  }
  return d;
}

// lambert plus ggx specular, schlick fresnel, smith shadowing, sky in the
// reflection.
// ambient: the light from everything but the sun (indirect_light()).
fn shade_material(m: Material, n: vec3f, v: vec3f, lit: f32, ao: f32, wrap: f32, coat: f32, ambient: vec3f) -> vec3f {
  let f0 = mix(vec3f(0.04), m.albedo, m.metallic);
  let nl = max(dot(n, SUN_DIR), 0.0);
  let nl_wrap = max((dot(n, SUN_DIR) + wrap) / (1.0 + wrap), 0.0);
  let nv = max(dot(n, v), 1e-4);
  let h = normalize(SUN_DIR + v);
  let nh = max(dot(n, h), 0.0);
  let vh = max(dot(v, h), 0.0);
  let a = m.roughness * m.roughness;
  let a2 = a * a;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let d = a2 / (3.14159 * dd * dd);
  let k = (m.roughness + 1.0) * (m.roughness + 1.0) / 8.0;
  let g = nl / (nl * (1.0 - k) + k) * nv / (nv * (1.0 - k) + k);
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
  let specular = 3.14159 * d * g * fresnel / max(4.0 * nl * nv, 1e-4);
  let diffuse = m.albedo * (1.0 - m.metallic) * (1.0 - fresnel);
  let env_f = f0 + (1.0 - f0) * pow(1.0 - nv, 5.0);
  let env = sky(reflect(-v, n)) * env_f * (1.0 - m.roughness) * 0.8 * ao;
  // the coat: ggx at roughness 0.05, fresnel from 0.04.
  let ca2 = 0.0025 * 0.0025;
  let cd = nh * nh * (ca2 - 1.0) + 1.0;
  let coat_f = coat * (0.04 + 0.96 * pow(1.0 - vh, 5.0));
  let coat_env = coat * (0.04 + 0.96 * pow(1.0 - nv, 5.0));
  let coat_spec = coat_f * ca2 / (4.0 * cd * cd * max(nv, 1e-4) * max(nl, 1e-4) + 1e-6) * nl;
  let base = (diffuse * nl_wrap + specular * nl) * SUN_COLOR * lit + m.albedo * (1.0 - m.metallic) * ambient + env +
             m.albedo * m.metallic * ambient * 0.5;
  return base * (1.0 - coat_env) + min(coat_spec, 50.0) * SUN_COLOR * lit + sky(reflect(-v, n)) * coat_env * ao;
}

// soft shadows, as vsm.glsl's: contact hardening (pcss) for a sun drawn 1.5 degrees wide.
const PENUMBRA_PER_UNIT = 0.0262;  // 2 tan(0.75 degrees)
const MAX_PENUMBRA = 12.0;         // texels, radius; marking covers it
const SEARCH_TAPS = 4u;
const FILTER_TAPS = 6u;

// stored depth at a texel (nearer layer); false without a physical page. taps mostly share the
// centre's page, whose physical page (home_phys) skips the entry read.
fn texel_depth(level: u32, at: vec2i, home: vec2i, home_phys: u32, home_flags: u32, stored: ptr<function, u32>) -> bool {
  let page = at >> vec2u(7u);
  var phys = home_phys;
  var flags = home_flags;
  if (any(page != home)) {
    if (!vsm_in_window(level, page)) { return false; }
    let slot = vsm_slot(level, page);
    if (vsm_entries[4u * slot] != vsm_tag(page)) { return false; }
    phys = vsm_entries[4u * slot + 1u];
    flags = vsm_entries[4u * slot + 3u];
  }
  if (phys == VSM_NONE) { return false; }
  let in_page = vec2u(at & vec2i(i32(VSM_PAGE) - 1));
  var s = vsm_atlas[vsm_atlas_index(phys, in_page, VSM_STILL)];
  if ((flags & VSM_HAS_MOVING) != 0u) { s = max(s, vsm_atlas[vsm_atlas_index(phys, in_page, VSM_MOVING)]); }
  *stored = s;
  return true;
}

// interleaved gradient noise, shifted per frame: taa averages the soft shadow taps.
fn shadow_noise(px: vec2u) -> f32 {
  return fract(52.9829189 * fract(dot(vec2f(px) + f32(frame.frame_index % 64u) * 5.588238, vec2f(0.06711056, 0.00583715))));
}

fn vogel(k: u32, n: u32, angle: f32) -> vec2f {
  let r = sqrt((f32(k) + 0.5) / f32(n));
  let a = f32(k) * 2.39996323 + angle;
  return r * vec2f(cos(a), sin(a));
}

// sunlight at p from the virtual shadow maps, 0 to 1, at the pixel's level: offset along the
// normal and biased a couple of texels; a blocker search sizes the penumbra, under a texel 2x2
// bilinear taps, else vogel taps rotated by `noise`. a tap without a physical page sends the
// lookup up a level.
fn sunlight(p: vec3f, n: vec3f, noise: f32) -> f32 {
  if ((frame.flags & FLAG_SHADOWS) == 0u) { return 1.0; }
  let soft = (frame.flags & FLAG_SOFT_SHADOWS) != 0u;
  for (var level = vsm_level_for_lit(length(p - frame.origin.xyz), dot(n, SUN_DIR)); level < VSM_LEVELS; level++) {
    let texel = vsm_texel(level);
    let lp = vsm_light_space(p + n * (2.0 * texel));
    let receiver = lp.z + 1.5 * texel;
    let centre = lp.xy / texel;
    let angle = noise * 6.2831853;
    let home = vec2i(floor(centre)) >> vec2u(7u);
    var home_phys = VSM_NONE;
    var home_flags = 0u;
    if (vsm_in_window(level, home)) {
      let slot = vsm_slot(level, home);
      if (vsm_entries[4u * slot] == vsm_tag(home)) {
        home_phys = vsm_entries[4u * slot + 1u];
        home_flags = vsm_entries[4u * slot + 3u];
      }
    }
    var complete = true;
    var blockers = 0.0;
    var count = 0.0;
    var stored = 0u;
    if (soft) {
      for (var k = 0u; k < SEARCH_TAPS; k++) {
        if (!texel_depth(level, vec2i(floor(centre + vogel(k, SEARCH_TAPS, angle) * MAX_PENUMBRA)), home, home_phys, home_flags, &stored)) {
          complete = false;
          break;
        }
        if (stored != 0u && vsm_unsortable(stored) > receiver) {
          blockers += vsm_unsortable(stored);
          count += 1.0;
        }
      }
      if (!complete) { continue; }
      if (count == 0.0) { return 1.0; }  // nothing nearby stands between it and the sun
    }
    var penumbra = 0.0;
    if (soft) { penumbra = (blockers / count - lp.z) * PENUMBRA_PER_UNIT * 0.5 / texel; }
    var lit = 0.0;
    if (penumbra < 1.0) {
      let f = centre - 0.5;
      let base = vec2i(floor(f));
      let w = f - vec2f(base);
      for (var k = 0; k < 4; k++) {
        if (!texel_depth(level, base + vec2i(k & 1, k >> 1u), home, home_phys, home_flags, &stored)) {
          complete = false;
          break;
        }
        let weight = select(1.0 - w.x, w.x, (k & 1) != 0) * select(1.0 - w.y, w.y, (k >> 1u) != 0);
        lit += select(0.0, weight, stored == 0u || vsm_unsortable(stored) <= receiver);
      }
    } else {
      let radius = min(penumbra, MAX_PENUMBRA);
      for (var k = 0u; k < FILTER_TAPS; k++) {
        if (!texel_depth(level, vec2i(floor(centre + vogel(k, FILTER_TAPS, angle + 1.0) * radius)), home, home_phys, home_flags, &stored)) {
          complete = false;
          break;
        }
        lit += select(0.0, 1.0, stored == 0u || vsm_unsortable(stored) <= receiver);
      }
      lit /= f32(FILTER_TAPS);
    }
    if (complete) { return lit; }
  }
  return 1.0;
}

// the ray from the camera through a point of the image (pixel units).
fn pixel_dir(p: vec2f) -> vec3f {
  let ndc = vec2f(p.x / f32(frame.width) * 2.0 - 1.0, 1.0 - p.y / f32(frame.height) * 2.0);
  let near_point = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  return normalize(near_point.xyz / near_point.w - frame.origin.xyz);
}

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= frame.width || gid.y >= frame.height) { return; }
  let px = vec2i(gid.xy);
  let ndc = vec2f((f32(gid.x) + 0.5) / f32(frame.width) * 2.0 - 1.0, 1.0 - (f32(gid.y) + 0.5) / f32(frame.height) * 2.0);
  let near_point = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  let origin = frame.origin.xyz;
  let dir = normalize(near_point.xyz / near_point.w - origin);

  let hd = textureLoad(hw_depth, px, 0);
  let sd_bits = atomicLoad(&sw_buf[gid.x + gid.y * frame.width]);
  let sd = bitcast<f32>(sd_bits);
  var found = false;
  var software = false;
  var z = 0.0;  // depth drawn, reversed
  var vc = vec2u(0u);
  var tri = 0u;
  var id = 0u;
  if (hd > 0.0 && hd >= sd) {
    id = textureLoad(hw_id, px, 0).x;
    vc = hw_visible[id >> 7u];
    found = true;
    z = hd;
  } else if (sd_bits != 0u) {
    id = atomicLoad(&sw_buf[frame.width * frame.height + gid.x + gid.y * frame.width]);
    vc = sw_visible[id >> 7u];
    found = true;
    software = true;
    z = sd;
  }
  tri = id & 127u;

  var color = sky(dir);
  if (!found && frame.debug_mode == 7u) {
    color = vec3f(1.0, 0.0, 1.0);  // holes: nothing drawn
  } else if (!found) {
    if (dir.y < 0.0) {
      let t = -origin.y / dir.y;
      let hit = origin + dir * t;
      let cell = abs(fract(hit.xz * 0.5) - 0.5);
      let line = smoothstep(0.47, 0.5, max(cell.x, cell.y));
      let ground = mix(vec3f(0.42, 0.4, 0.37), vec3f(0.33, 0.31, 0.29), line) / GROUND_ALBEDO *
                   ground_radiance(sunlight(hit, vec3f(0.0, 1.0, 0.0), shadow_noise(gid.xy)), occlusion(gid.xy));
      color = mix(ground, color, 1.0 - exp(-t * 0.012));
    }
  } else {
    let inst = load_instance(vc.x);
    let c = load_cluster(vc.y);
    let packed = cluster_triangle(c, tri);
    let i0 = packed & 255u;
    let i1 = (packed >> 8u) & 255u;
    let i2 = (packed >> 16u) & 255u;
    let grid = meshes[inst.mesh].grid;
    let bounds = meshes[inst.mesh].bounds;
    let p0 = to_world(inst, drawn_vertex(inst, c, i0));
    let p1 = to_world(inst, drawn_vertex(inst, c, i1));
    let p2 = to_world(inst, drawn_vertex(inst, c, i2));
    // ray against the triangle's plane, for exact barycentrics.
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let pv = cross(dir, e2);
    let det = dot(e1, pv);
    let inv = select(0.0, 1.0 / det, abs(det) > 1e-20);
    let tv = origin - p0;
    let qv = cross(tv, e1);
    // clamped: at a silhouette the triangle can be edge-on and the plane hit
    // far outside.
    let bu = clamp(dot(tv, pv) * inv, 0.0, 1.0);
    let bv = clamp(dot(dir, qv) * inv, 0.0, 1.0 - bu);
    // distance from depth, which is exact.
    let center = frame.inv_view_proj * vec4f(0.0, 0.0, 1.0, 1.0);
    let forward = normalize(center.xyz / center.w - origin);
    let t = frame.near_z / z / dot(dir, forward);
    var n0 = cluster_normal(c, i0);
    var n1 = cluster_normal(c, i1);
    var n2 = cluster_normal(c, i2);
    if (skinned(inst)) {  // each vertex's normal turned by its joints
      n0 = skin_dir(inst, c, i0, n0);
      n1 = skin_dir(inst, c, i1, n1);
      n2 = skin_dir(inst, c, i2, n2);
    }
    let model_n = deform_normal(inst, bounds, from_world(inst, origin + dir * t), n0 * (1.0 - bu - bv) + n1 * bu + n2 * bv, frame.time);
    var n = normalize(to_world_dir(inst, model_n));
    // from behind (inside a fold, through a scan hole): light the visible side,
    // no rim light, which would outline every sliver.
    let behind = dot(n, dir) > 0.0;
    if (behind) { n = -n; }

    var m = material(inst.material);
    let ao = occlusion(gid.xy);
    let geometric_n = n;
    var wrap = 0.0;
    var coat = 0.0;
    let tex = meshes[inst.mesh].tex;
    if (frame.debug_mode == 0u && tex.y != 0u && cluster_textured(c)) {
      // a scan's own colour: no procedural detail. texture coordinates where this pixel's ray
      // and its neighbours' meet the triangle's plane: their differences pick the level.
      let uv0 = cluster_uv(c, i0);
      let uv1 = cluster_uv(c, i1);
      let uv2 = cluster_uv(c, i2);
      let pf = vec2f(gid.xy) + 0.5;
      let b0 = plane_bary(p0, p1, p2, origin, dir);
      let bx = plane_bary(p0, p1, p2, origin, pixel_dir(pf + vec2f(1.0, 0.0)));
      let by = plane_bary(p0, p1, p2, origin, pixel_dir(pf + vec2f(0.0, 1.0)));
      let uv = uv0 + (uv1 - uv0) * bu + (uv2 - uv0) * bv;
      let duv_dx = (uv1 - uv0) * (bx.x - b0.x) + (uv2 - uv0) * (bx.y - b0.y);
      let duv_dy = (uv1 - uv0) * (by.x - b0.x) + (uv2 - uv0) * (by.y - b0.y);
      m = Material(sample_texture(tex, uv, duv_dx, duv_dy), 0.75, 0.0);
    } else if (frame.debug_mode == 0u) {
      let radius = bounds.w;
      // detail is fixed to the rest surface: a swaying statue's grain bends with it.
      let drawn = from_world(inst, origin + dir * t);
      // skinned: no inverse, the pixel's weights on the triangle's rest corners.
      var rest = undeform(inst, bounds, drawn, frame.time);
      if (skinned(inst)) {
        rest = cluster_position(c, grid, i0) * (1.0 - bu - bv) + cluster_position(c, grid, i1) * bu + cluster_position(c, grid, i2) * bv;
      }
      let q = rest / radius;
      let d = surface_detail(m, min(inst.material, 5u), q, t / (frame.lod_scale * inst.scale * radius), ao, n.y);
      m = d.m;
      wrap = d.wrap;
      coat = d.coat;
      var model_g = deform_gradient(inst, bounds, drawn, d.grad, frame.time);
      if (skinned(inst)) { model_g = skin_dir(inst, c, i0, d.grad); }
      let g = to_world_dir(inst, model_g) / inst.scale;
      n = normalize(n - (g - dot(g, n) * n));
    }
    if (frame.debug_mode != 0u) { m = Material(vec3f(0.56, 0.52, 0.47), 0.6, 0.0); }
    switch (frame.debug_mode) {
      case 1u: { m.albedo = hash_color(vc.y * 7919u + vc.x * 104729u); }
      case 2u: { m.albedo = hash_color(id * 2654435761u + vc.x + select(0u, 0x9e3779b9u, software)); }
      case 3u: { m.albedo = level_color(cluster_level(c)); }
      case 4u: { m.albedo = hash_color(c.group + vc.x * 104729u); }
      case 5u: { m.albedo = hash_color(vc.x); }
      case 6u: { m.albedo = select(vec3f(0.15, 0.45, 0.95), vec3f(0.95, 0.45, 0.1), software); }
      case 7u: { m.albedo = vec3f(0.3); }
      default: {}
    }
    let hit = origin + dir * t;
    let lit = select(0.0, sunlight(hit, geometric_n, shadow_noise(gid.xy)), dot(geometric_n, SUN_DIR) > 0.0);
    color = shade_material(m, n, -dir, lit, ao, wrap, coat, indirect_light(gid.xy, n));
    if (!behind) { color += pow(1.0 - max(dot(n, -dir), 0.0), 4.0) * 0.25 * sky(n) * (1.0 - m.metallic) * ao; }
    color = mix(color, sky(dir), 1.0 - exp(-t * 0.012));
  }
  if (frame.debug_mode == 8u && (found || dir.y < 0.0)) { color = vec3f(occlusion(gid.xy)); }  // occlusion alone
  textureStore(out_image, px, vec4f(srgb(aces(color)), 1.0));
}

// taa, as taa.comp: each frame jittered, each pixel blended 10% into last
// frame's image, found by projecting its surface with last frame's camera,
// clamped to this frame's 3x3 neighbourhood. history at 16 bits: 10% steps
// stall in 8.
@group(2) @binding(8) var shaded: texture_2d<f32>;
@group(2) @binding(9) var history: texture_2d<f32>;
@group(2) @binding(10) var history_sampler: sampler;
@group(2) @binding(11) var history_out: texture_storage_2d<rgba16float, write>;
@group(2) @binding(12) var display: texture_storage_2d<rgba8unorm, write>;

fn to_ycocg(c: vec3f) -> vec3f {
  return vec3f(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

fn from_ycocg(c: vec3f) -> vec3f {
  return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// last frame's image through catmull-rom in five bilinear taps, so motion does
// not blur history.
fn history_at(uv: vec2f) -> vec3f {
  let size = vec2f(f32(frame.width), f32(frame.height));
  let p = uv * size;
  let center = floor(p - 0.5) + 0.5;
  let f = p - center;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = -0.5 * f3 + f2 - 0.5 * f;
  let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
  let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
  let w3 = 0.5 * f3 - 0.5 * f2;
  let w12 = w1 + w2;
  let t0 = (center - 1.0) / size;
  let t3 = (center + 2.0) / size;
  let t12 = (center + w2 / w12) / size;
  var c = textureSampleLevel(history, history_sampler, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
  c += textureSampleLevel(history, history_sampler, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
  c += textureSampleLevel(history, history_sampler, t12, 0.0).rgb * (w12.x * w12.y);
  c += textureSampleLevel(history, history_sampler, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
  c += textureSampleLevel(history, history_sampler, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
  let w = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return max(c / w, vec3f(0.0));
}

@compute @workgroup_size(8, 8)
fn taa(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= frame.width || gid.y >= frame.height) { return; }
  let px = vec2i(gid.xy);
  let c = textureLoad(shaded, px, 0).rgb;
  var out = c;
  if (frame.taa_valid != 0u && frame.debug_mode == 0u) {
    var lo = to_ycocg(c);
    var hi = lo;
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let q = clamp(px + vec2i(x, y), vec2i(0), vec2i(i32(frame.width), i32(frame.height)) - 1);
        let n = to_ycocg(textureLoad(shaded, q, 0).rgb);
        lo = min(lo, n);
        hi = max(hi, n);
      }
    }
    let ndc = vec2f((f32(gid.x) + 0.5) / f32(frame.width) * 2.0 - 1.0, 1.0 - (f32(gid.y) + 0.5) / f32(frame.height) * 2.0);
    let near_point = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
    let origin = frame.origin.xyz;
    let dir = normalize(near_point.xyz / near_point.w - origin);
    let hd = textureLoad(hw_depth, px, 0);
    let sd_bits = atomicLoad(&sw_buf[gid.x + gid.y * frame.width]);
    let z = max(hd, bitcast<f32>(sd_bits));
    var prev: vec4f;
    if (z > 0.0) {
      let center = frame.inv_view_proj * vec4f(0.0, 0.0, 1.0, 1.0);
      let forward = normalize(center.xyz / center.w - origin);
      var p = origin + dir * (frame.near_z / z / dot(dir, forward));
      // a moving surface was elsewhere last frame: through its instance.
      var vc = vec2u(0u);
      var id = 0u;
      if (hd > 0.0 && hd >= bitcast<f32>(sd_bits)) {
        id = textureLoad(hw_id, px, 0).x;
        vc = hw_visible[id >> 7u];
      } else {
        id = atomicLoad(&sw_buf[frame.width * frame.height + gid.x + gid.y * frame.width]);
        vc = sw_visible[id >> 7u];
      }
      let instance = vc.x;
      let inst = load_instance(instance);
      if (skinned(inst)) {
        // no inverse: the pixel's weights on its triangle as drawn now, applied to the
        // triangle's corners posed as last frame.
        let c = load_cluster(vc.y);
        let tri = cluster_triangle(c, id & 127u);
        let i0 = tri & 255u;
        let i1 = (tri >> 8u) & 255u;
        let i2 = (tri >> 16u) & 255u;
        let grid = meshes[inst.mesh].grid;
        let r0 = cluster_position(c, grid, i0);
        let r1 = cluster_position(c, grid, i1);
        let r2 = cluster_position(c, grid, i2);
        let a = to_world(inst, skin_point(inst, c, i0, r0, false));
        let b = to_world(inst, skin_point(inst, c, i1, r1, false));
        let e = to_world(inst, skin_point(inst, c, i2, r2, false));
        let w = plane_bary(a, b, e, origin, dir);
        let then = skin_point(inst, c, i0, r0, true) * (1.0 - w.x - w.y) + skin_point(inst, c, i1, r1, true) * w.x +
                   skin_point(inst, c, i2, r2, true) * w.y;
        p = to_world(load_prev_instance(instance), then);
      } else if (inst.anim != 0u) {
        let bounds = meshes[inst.mesh].bounds;
        let rest = undeform(inst, bounds, from_world(inst, p), frame.time);
        p = to_world(load_prev_instance(instance), deform(inst, bounds, rest, frame.prev_time));
      }
      prev = frame.prev_view_proj * vec4f(p, 1.0);
    } else if (dir.y < 0.0) {
      prev = frame.prev_view_proj * vec4f(origin + dir * (-origin.y / dir.y), 1.0);
    } else {
      prev = frame.prev_view_proj * vec4f(dir, 0.0);  // sky: a direction at infinity
    }
    // found along the nudged ray; removing the nudge lets a still camera read
    // history at pixel centres.
    let p = prev.xy / prev.w + frame.jitter;
    let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    if (prev.w > 0.0 && all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0))) {
      let h = from_ycocg(clamp(to_ycocg(history_at(uv)), lo, hi));
      out = mix(h, c, 0.1);
    }
  }
  textureStore(history_out, px, vec4f(out, 1.0));
  textureStore(display, px, vec4f(out, 1.0));
}
