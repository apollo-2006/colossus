// virtual shadow map passes (vsm_common.wgsl), a module of their own to keep
// the bind group small. per frame: mark needed pages, invalidate what moving
// instances crossed, assign physical pages, clear and render the listed ones
// from the hierarchy seen from the sun.

@group(1) @binding(0) var<storage, read_write> entries: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> lists: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> atlas: array<atomic<u32>>;
// vwork: [0] work count, [1] big count, [2] visible count, then work items, big
// instances, visible clusters (VW_*).
@group(1) @binding(3) var<storage, read_write> vwork: array<atomic<u32>>;
@group(1) @binding(4) var hw_depth: texture_depth_2d;
@group(1) @binding(5) var<storage, read> sw_depth: array<u32>;
@group(1) @binding(6) var<storage, read_write> stamps: array<atomic<u32>>;    // compute.wgsl's page_stamps
@group(1) @binding(7) var<storage, read_write> request_words: array<atomic<u32>>;  // compute.wgsl's requests
// indirect arguments, bound only for the passes writing them: a dispatch cannot
// write its argument buffer.
@group(2) @binding(0) var<storage, read_write> vargs: array<u32>;

const VA_INSTANCE = 0u;  // into vargs
const VA_CLEAR = 4u;
const VA_EXPAND = 8u;
const VA_CULL = 12u;
const VA_RASTER = 16u;
const VW_WORK = 4u;
const VSM_MAX_WORK = 1048576u;
const VW_BIG = 2097156u;     // VW_WORK + 2 * VSM_MAX_WORK
const VSM_BIG_CAPACITY = 65536u;
const VW_VISIBLE = 2359300u; // VW_BIG + 4 * VSM_BIG_CAPACITY
const VSM_MAX_VISIBLE = 1048576u;
const BIG_PIECES = 16u;

fn word(k: u32) -> u32 { return atomicLoad(&vwork[k]); }

fn renders(layer: u32, slot: u32) -> bool {
  return (atomicLoad(&lists[vsm_mask_word(layer, slot)]) & (1u << (slot % VSM_WINDOW))) != 0u;
}

fn rendered_rect(layer: u32, level: u32) -> vec4i {
  let b = VSM_RECTS + (layer * VSM_LEVELS + level) * 4u;
  return vec4i((1 << 20u) - i32(atomicLoad(&lists[b])), (1 << 20u) - i32(atomicLoad(&lists[b + 1u])),
               i32(atomicLoad(&lists[b + 2u])) - (1 << 20u), i32(atomicLoad(&lists[b + 3u])) - (1 << 20u));
}

fn touches(layer: u32, level: u32, center: vec2f, radius: f32) -> bool {
  let size = vsm_texel(level) * f32(VSM_PAGE);
  let r = rendered_rect(layer, level);
  let lo = max(vec2i(floor((center - radius) / size)), r.xy);
  let hi = min(vec2i(floor((center + radius) / size)), r.zw);
  if (any(lo > hi)) { return false; }
  if (hi.x - lo.x > 7 || hi.y - lo.y > 7) { return true; }
  for (var y = lo.y; y <= hi.y; y++) {
    for (var x = lo.x; x <= hi.x; x++) {
      let page = vec2i(x, y);
      if (vsm_in_window(level, page) && renders(layer, vsm_slot(level, page))) { return true; }
    }
  }
  return false;
}

// --- marking: a 2x2 block per invocation. each pixel marks its plain level; where the block is
// one smooth surface, its normal from the four points marks the grazing level too.
fn surface_point(px_in: vec2i, t: ptr<function, f32>) -> vec3f {
  let px = vec2u(clamp(px_in, vec2i(0), vec2i(i32(frame.width), i32(frame.height)) - 1));
  let ndc = vec2f((f32(px.x) + 0.5) / f32(frame.width) * 2.0 - 1.0, 1.0 - (f32(px.y) + 0.5) / f32(frame.height) * 2.0);
  let near_point = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  let origin = frame.origin.xyz;
  let dir = normalize(near_point.xyz / near_point.w - origin);
  let z = max(textureLoad(hw_depth, px, 0), bitcast<f32>(sw_depth[px.x + px.y * frame.width]));
  *t = -1.0;
  if (z > 0.0) {
    let center = frame.inv_view_proj * vec4f(0.0, 0.0, 1.0, 1.0);
    let forward = normalize(center.xyz / center.w - origin);
    *t = frame.near_z / z / dot(dir, forward);
  } else if (dir.y < 0.0) {
    *t = -origin.y / dir.y;
  }
  return origin + dir * *t;
}

fn mark(level: u32, p: vec3f) {
  let texel = vsm_light_space(p).xy / vsm_texel(level);
  // the lookup offsets a couple of texels and searches 12 around (compute.wgsl's MAX_PENUMBRA).
  let reach = 15.0;
  for (var k = 0u; k < 4u; k++) {
    let corner = texel + vec2f(f32(k & 1u), f32(k >> 1u)) * (2.0 * reach) - reach;
    let page = vec2i(floor(corner / f32(VSM_PAGE)));
    if (vsm_in_window(level, page)) { atomicStore(&entries[4u * vsm_slot(level, page) + 2u], frame.frame_index); }
  }
}

@compute @workgroup_size(8, 8)
fn vsm_mark(@builtin(global_invocation_id) gid: vec3u) {
  let block = vec2i(gid.xy) * 2;
  if (block.x >= i32(frame.width) || block.y >= i32(frame.height)) { return; }
  var p: array<vec3f, 4>;
  var t: array<f32, 4>;
  for (var k = 0; k < 4; k++) {
    var tk = 0.0;
    p[k] = surface_point(block + vec2i(k & 1, k >> 1u), &tk);
    t[k] = tk;
    if (tk >= 0.0) { mark(vsm_level_for(tk), p[k]); }
  }
  for (var k = 0; k < 4; k++) {
    if (t[k] < 0.0 || abs(t[k] - t[0]) > 0.02 * t[0]) { return; }
  }
  var n = normalize(cross(p[1] - p[0], p[2] - p[0]));
  if (dot(n, p[0] - frame.origin.xyz) > 0.0) { n = -n; }
  let nl = dot(n, VSM_SUN);
  if (nl <= 0.0) { return; }
  for (var k = 0; k < 4; k++) {
    let level = vsm_level_for_lit(t[k], nl);
    if (level != vsm_level_for(t[k])) { mark(level, p[k]); }
  }
}

// --- allocation.
fn render(slot: u32, layer: u32) {
  atomicStore(&lists[VSM_RENDER + atomicAdd(&lists[3], 1u)], slot | (layer << 31u));
  let level = slot / (VSM_WINDOW * VSM_WINDOW);
  atomicOr(&lists[vsm_mask_word(layer, slot)], 1u << (slot % VSM_WINDOW));
  atomicOr(&lists[4u + layer], 1u << level);
  let biased = vsm_slot_page(slot) + vec2i(1 << 20u);
  let b = VSM_RECTS + (layer * VSM_LEVELS + level) * 4u;
  atomicMax(&lists[b], u32((1 << 21u) - biased.x));
  atomicMax(&lists[b + 1u], u32((1 << 21u) - biased.y));
  atomicMax(&lists[b + 2u], u32(biased.x));
  atomicMax(&lists[b + 3u], u32(biased.y));
}

fn render_new(slot: u32) {
  render(slot, VSM_STILL);
  if ((frame.flags & FLAG_MOVING) != 0u) { render(slot, VSM_MOVING); }
}

fn invalidate(world_center: vec3f, radius: f32, lane: u32) {
  let center = vsm_light_space(world_center).xy;
  let half = vec2i(i32(VSM_WINDOW / 2u));
  for (var level = 0u; level < VSM_LEVELS; level++) {
    let size = vsm_texel(level) * f32(VSM_PAGE);
    let c = vsm_center(level);
    let lo = max(vec2i(floor((center - radius) / size)), c - half);
    let hi = min(vec2i(floor((center + radius) / size)), c + half - 1);
    if (any(lo > hi)) { continue; }
    let width = u32(hi.x - lo.x + 1);
    let count = width * u32(hi.y - lo.y + 1);
    for (var k = lane; k < count; k += 64u) {
      let page = lo + vec2i(i32(k % width), i32(k / width));
      let slot = vsm_slot(level, page);
      if (atomicLoad(&entries[4u * slot + 1u]) != VSM_NONE && atomicLoad(&entries[4u * slot]) == vsm_tag(page)) {
        atomicOr(&entries[4u * slot + 3u], VSM_DIRTY);
      }
    }
  }
}

// a workgroup per moving instance: where it was, where it is.
@compute @workgroup_size(64)
fn vsm_invalidate(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  let g = wid.y * 65535u + wid.x;
  if (VSM_MOVING_LIST + g >= arrayLength(&lists)) { return; }
  let i = atomicLoad(&lists[VSM_MOVING_LIST + g]);
  let now = load_instance(i);
  let then = load_prev_instance(i);
  let m = meshes[now.mesh];
  invalidate(to_world(then, m.bounds.xyz), m.bounds.w * now.scale, lane);
  invalidate(to_world(now, m.bounds.xyz), m.bounds.w * now.scale, lane);
}

@compute @workgroup_size(64)
fn vsm_alloc_slots(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= VSM_SLOTS) { return; }
  let phys = atomicLoad(&entries[4u * k + 1u]);
  let resident = phys != VSM_NONE && atomicLoad(&entries[4u * k]) == vsm_tag(vsm_slot_page(k));
  if (phys != VSM_NONE && !resident) {
    atomicStore(&entries[VSM_PHYS + 2u * phys], VSM_NONE);
    atomicStore(&entries[4u * k + 1u], VSM_NONE);
  }
  if (atomicLoad(&entries[4u * k + 2u]) != frame.frame_index) { return; }
  if (resident) {
    atomicStore(&entries[VSM_PHYS + 2u * phys + 1u], frame.frame_index);
    let flags = atomicAnd(&entries[4u * k + 3u], ~(VSM_DIRTY | VSM_PROVISIONAL));
    if ((flags & VSM_PROVISIONAL) != 0u) { render(k, VSM_STILL); }
    if ((flags & VSM_DIRTY) != 0u) { render(k, VSM_MOVING); }
  } else {
    atomicStore(&lists[VSM_REQUESTS + atomicAdd(&lists[0], 1u)], k);
  }
}

@compute @workgroup_size(64)
fn vsm_alloc_phys(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= VSM_SIDE * VSM_SIDE) { return; }
  if (atomicLoad(&entries[VSM_PHYS + 2u * p + 1u]) == frame.frame_index) { return; }
  // free first, then unneeded longest: pages needed every few frames (taa's jitter moves the
  // marked edge) would otherwise be evicted and redrawn over and over.
  if (atomicLoad(&entries[VSM_PHYS + 2u * p]) == VSM_NONE) {
    atomicStore(&lists[VSM_UNOWNED + atomicAdd(&lists[1], 1u)], p);
  } else if (frame.frame_index - atomicLoad(&entries[VSM_PHYS + 2u * p + 1u]) >= VSM_STALE) {
    atomicStore(&lists[VSM_EVICTABLE + atomicAdd(&lists[2], 1u)], p);
  } else {
    atomicStore(&lists[VSM_EVICTABLE + VSM_SLOTS / 2u + atomicAdd(&lists[7], 1u)], p);
  }
}

@compute @workgroup_size(64)
fn vsm_alloc_assign(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= atomicLoad(&lists[0])) { return; }
  let unowned = atomicLoad(&lists[1]);
  let stale = atomicLoad(&lists[2]);
  if (k >= unowned + stale + atomicLoad(&lists[7])) {
    atomicAdd(&lists[6], 1u);
    return;
  }
  var p = 0u;
  if (k < unowned) {
    p = atomicLoad(&lists[VSM_UNOWNED + k]);
  } else if (k < unowned + stale) {
    p = atomicLoad(&lists[VSM_EVICTABLE + k - unowned]);
  } else {
    p = atomicLoad(&lists[VSM_EVICTABLE + VSM_SLOTS / 2u + k - unowned - stale]);
  }
  let old = atomicLoad(&entries[VSM_PHYS + 2u * p]);
  if (old != VSM_NONE) { atomicStore(&entries[4u * old + 1u], VSM_NONE); }
  let slot = atomicLoad(&lists[VSM_REQUESTS + k]);
  atomicStore(&entries[4u * slot], vsm_tag(vsm_slot_page(slot)));
  atomicStore(&entries[4u * slot + 1u], p);
  atomicStore(&entries[4u * slot + 3u], 0u);
  atomicStore(&entries[VSM_PHYS + 2u * p], slot);
  atomicStore(&entries[VSM_PHYS + 2u * p + 1u], frame.frame_index);
  render_new(slot);
}

// --- indirect arguments.
fn rows(n: u32, at: u32) {
  vargs[at] = min(n, 65535u);
  vargs[at + 1u] = (n + 65534u) / 65535u;
  vargs[at + 2u] = 1u;
}

@compute @workgroup_size(1)
fn vsm_args_alloc() {
  let any = (atomicLoad(&lists[4]) | atomicLoad(&lists[5])) != 0u;
  rows(select(0u, (frame.instance_count + 63u) / 64u, any), VA_INSTANCE);
  rows(atomicLoad(&lists[3]), VA_CLEAR);
}

@compute @workgroup_size(1)
fn vsm_args_expand() { rows(min(word(1), VSM_BIG_CAPACITY), VA_EXPAND); }

@compute @workgroup_size(1)
fn vsm_args_cull() { rows(min(word(0), VSM_MAX_WORK), VA_CULL); }

@compute @workgroup_size(1)
fn vsm_args_raster() { rows(min(word(2), VSM_MAX_VISIBLE), VA_RASTER); }

// --- clearing: a workgroup per page and layer.
@compute @workgroup_size(256)
fn vsm_clear(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  let k = wid.y * 65535u + wid.x;
  if (k >= atomicLoad(&lists[3])) { return; }
  let e = atomicLoad(&lists[VSM_RENDER + k]);
  let phys = atomicLoad(&entries[4u * (e & 0x7fffffffu) + 1u]);
  // an empty moving layer is skipped by lookups until vsm_raster draws into it.
  if ((e >> 31u) == VSM_MOVING && lane == 0u) { atomicAnd(&entries[4u * (e & 0x7fffffffu) + 3u], ~VSM_HAS_MOVING); }
  for (var t = lane; t < VSM_PAGE * VSM_PAGE; t += 256u) {
    atomicStore(&atlas[vsm_atlas_index(phys, vec2u(t % VSM_PAGE, t / VSM_PAGE), e >> 31u)], 0u);
  }
}

// --- culling: per instance, each rendering level of its layer; errors in that
// level's texels.
fn emit(at: u32, tagged: u32, first: u32) {
  atomicStore(&vwork[VW_WORK + 2u * at], tagged);
  atomicStore(&vwork[VW_WORK + 2u * at + 1u], first);
}

@compute @workgroup_size(64)
fn vsm_instance(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.y * 65535u * 64u + gid.x;
  if (i >= frame.instance_count) { return; }
  let inst = load_instance(i);
  let m = meshes[inst.mesh];
  let center = vsm_light_space(to_world(inst, m.bounds.xyz)).xy;
  let radius = m.bounds.w * inst.scale;
  let layer = select(VSM_STILL, VSM_MOVING, inst.anim != 0u);
  var levels = atomicLoad(&lists[4u + layer]);
  while (levels != 0u) {
    let level = firstTrailingBit(levels);
    levels &= levels - 1u;
    if (!touches(layer, level, center, radius)) { continue; }
    let limit = frame.lod_threshold * vsm_texel(level) / inst.scale;
    var lo = 0u;
    var hi = m.cluster_count;
    while (lo < hi) {
      let mid = (lo + hi) / 2u;
      if (clusters[m.first_cluster + mid].parent_error > limit) { hi = mid; } else { lo = mid + 1u; }
    }
    let chunks = (m.cluster_count - lo + 63u) / 64u;
    if (chunks == 0u) { continue; }
    let tagged = i | (level << 24u) | (layer << 28u);
    if (chunks > BIG_PIECES) {
      let b = atomicAdd(&vwork[1], 1u);
      if (b < VSM_BIG_CAPACITY) {
        atomicStore(&vwork[VW_BIG + 4u * b], tagged);
        atomicStore(&vwork[VW_BIG + 4u * b + 1u], m.first_cluster + lo);
        atomicStore(&vwork[VW_BIG + 4u * b + 2u], chunks);
      }
      continue;
    }
    let base = atomicAdd(&vwork[0], chunks);
    for (var c = 0u; c < chunks && base + c < VSM_MAX_WORK; c++) { emit(base + c, tagged, m.first_cluster + lo + 64u * c); }
  }
}

var<workgroup> wg_base: u32;
var<workgroup> wg_count: u32;

@compute @workgroup_size(64)
fn vsm_expand(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  let k = wid.y * 65535u + wid.x;
  if (lane == 0u) {
    wg_count = 0u;
    if (k < min(word(1), VSM_BIG_CAPACITY)) {
      let chunks = word(VW_BIG + 4u * k + 2u);
      let b = atomicAdd(&vwork[0], chunks);
      wg_base = b;
      wg_count = select(0u, min(chunks, VSM_MAX_WORK - b), b < VSM_MAX_WORK);
    }
  }
  let count = workgroupUniformLoad(&wg_count);
  let base = workgroupUniformLoad(&wg_base);
  if (count == 0u) { return; }
  let tagged = word(VW_BIG + 4u * k);
  let first = word(VW_BIG + 4u * k + 1u);
  for (var c = lane; c < count; c += 64u) { emit(base + c, tagged, first + 64u * c); }
}

fn request(c: Cluster, priority: f32) {
  let stamp = arrayLength(&stamps) / 2u + c.creator;
  if (atomicExchange(&stamps[stamp], frame.frame_index) == frame.frame_index) { return; }
  let k = atomicAdd(&request_words[0], 1u);
  if (k < frame.max_requests) {
    atomicStore(&request_words[4u + 2u * k], c.creator);
    atomicStore(&request_words[4u + 2u * k + 1u], bitcast<u32>(priority));
  }
}

@compute @workgroup_size(64)
fn vsm_cluster(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  let item = wid.y * 65535u + wid.x;
  if (item >= min(word(0), VSM_MAX_WORK)) { return; }
  let tagged = word(VW_WORK + 2u * item);
  let instance = tagged & 0xffffffu;
  let level = (tagged >> 24u) & 15u;
  let layer = tagged >> 28u;
  let inst = load_instance(instance);
  let m = meshes[inst.mesh];
  let cluster = word(VW_WORK + 2u * item + 1u) + lane;
  if (cluster >= m.first_cluster + m.cluster_count) { return; }
  let c = clusters[cluster];
  let texel = vsm_texel(level);
  let self_error = c.lod_error * inst.scale / texel;
  let coarse_enough = c.parent_error * inst.scale / texel > frame.lod_threshold;
  let finer_resident = c.creator != NO_PAGE && page_table[c.creator] != NO_PAGE;
  let wants_finer = self_error > frame.lod_threshold && c.creator != NO_PAGE && !finer_resident;
  let loaded = page_table[c.group] != NO_PAGE;
  if (!coarse_enough || (self_error > frame.lod_threshold && finer_resident)) { return; }
  let center = vsm_light_space(to_world(inst, c.center)).xy;
  let radius = c.radius * inst.scale;
  if (!touches(layer, level, center, radius)) { return; }
  // a stand-in, or unloaded: its pages render again next frame (the moving
  // layer does anyway).
  if ((wants_finer || !loaded) && layer == VSM_STILL) {
    let size = texel * f32(VSM_PAGE);
    let r = rendered_rect(layer, level);
    let lo = max(vec2i(floor((center - radius) / size)), r.xy);
    let hi = min(vec2i(floor((center + radius) / size)), r.zw);
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        if (vsm_in_window(level, vec2i(x, y))) { atomicOr(&entries[4u * vsm_slot(level, vec2i(x, y)) + 3u], VSM_PROVISIONAL); }
      }
    }
  }
  if (!loaded) { return; }
  atomicStore(&stamps[c.group], frame.frame_index);
  if (wants_finer) { request(c, self_error); }
  let k = atomicAdd(&vwork[2], 1u);
  if (k < VSM_MAX_VISIBLE) {
    atomicStore(&vwork[VW_VISIBLE + 2u * k], tagged);
    atomicStore(&vwork[VW_VISIBLE + 2u * k + 1u], cluster);
  }
}

// --- rasterizing, as vsm_raster.comp: level texels snapped to 1/16, edges from
// each triangle's corner, top-left rule, both sides, atomic max per texel in
// its page. the walk is clipped to the level's rendered pages and skips pages
// not rendering; a triangle over 32x32 texels after clipping is drawn by the
// whole workgroup, a texel an invocation.
var<workgroup> snapped: array<vec2i, 128>;
var<workgroup> depth: array<f32, 128>;
var<workgroup> together: array<u32, 128>;  // triangles for the whole workgroup
var<workgroup> together_count: atomic<u32>;

fn edge(a: vec2i, b: vec2i, p: vec2i) -> i32 {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn owns_edge(a: vec2i, b: vec2i) -> bool {
  let d = b - a;
  return d.y < 0 || (d.y == 0 && d.x > 0);
}

// corners relative to lo_abs * 16, wound positive; the clipped box of texels, [start, hi] from lo_abs.
struct Tri {
  a: vec2i, b: vec2i, d: vec2i,
  za: f32, zb: f32, zd: f32,
  lo_abs: vec2i, start: vec2i, hi: vec2i,
  area: i32, bias0: i32, bias1: i32, bias2: i32,
  ok: bool,
}

fn setup(packed: u32, level: u32, layer: u32) -> Tri {
  var tri: Tri;
  tri.ok = false;
  let i0 = packed & 255u;
  let i1 = (packed >> 8u) & 255u;
  let i2 = (packed >> 16u) & 255u;
  tri.a = snapped[i0];
  tri.b = snapped[i1];
  tri.d = snapped[i2];
  tri.za = depth[i0];
  tri.zb = depth[i1];
  tri.zd = depth[i2];
  tri.lo_abs = (min(tri.a, min(tri.b, tri.d)) - 8 + 15) >> vec2u(4u);
  let hi_abs = (max(tri.a, max(tri.b, tri.d)) - 8) >> vec2u(4u);
  if (any(tri.lo_abs > hi_abs) || any(hi_abs - tri.lo_abs > vec2i(1023))) { return tri; }
  let origin = tri.lo_abs * 16;
  tri.a -= origin;
  tri.b -= origin;
  tri.d -= origin;
  tri.area = edge(tri.a, tri.b, tri.d);
  if (tri.area == 0) { return tri; }
  if (tri.area < 0) {
    let tb = tri.b; tri.b = tri.d; tri.d = tb;
    let tz = tri.zb; tri.zb = tri.zd; tri.zd = tz;
    tri.area = -tri.area;
  }
  let rect = rendered_rect(layer, level);
  let page = i32(VSM_PAGE);
  tri.start = max(tri.lo_abs, rect.xy * page) - tri.lo_abs;
  tri.hi = min(hi_abs, rect.zw * page + page - 1) - tri.lo_abs;
  if (any(tri.start > tri.hi)) { return tri; }
  tri.bias0 = select(-1, 0, owns_edge(tri.b, tri.d));
  tri.bias1 = select(-1, 0, owns_edge(tri.d, tri.a));
  tri.bias2 = select(-1, 0, owns_edge(tri.a, tri.b));
  tri.ok = true;
  return tri;
}

// the page's physical page if it renders, else VSM_NONE.
fn rendering(level: u32, layer: u32, page: vec2i) -> u32 {
  if (!vsm_in_window(level, page)) { return VSM_NONE; }
  let slot = vsm_slot(level, page);
  if (!renders(layer, slot)) { return VSM_NONE; }
  return atomicLoad(&entries[4u * slot + 1u]);
}

fn store(phys: u32, layer: u32, at: vec2i, z: f32) {
  let index = vsm_atlas_index(phys, vec2u(at & vec2i(i32(VSM_PAGE) - 1)), layer);
  let value = vsm_sortable(z);
  if (value > atomicLoad(&atlas[index])) { atomicMax(&atlas[index], value); }
}

fn mark_moving(level: u32, layer: u32, page: vec2i) {
  if (layer == VSM_MOVING) { atomicOr(&entries[4u * vsm_slot(level, page) + 3u], VSM_HAS_MOVING); }
}

fn depth_at(tri: Tri, w0: i32, w1: i32, w2: i32) -> f32 {
  return (f32(w0 - tri.bias0) * tri.za + f32(w1 - tri.bias1) * tri.zb + f32(w2 - tri.bias2) * tri.zd) / f32(tri.area);
}

// one invocation: rows, a page's span at a time.
fn walk(tri: Tri, level: u32, layer: u32) {
  let p0 = tri.start * 16 + 8;
  var row0 = edge(tri.b, tri.d, p0) + tri.bias0;
  var row1 = edge(tri.d, tri.a, p0) + tri.bias1;
  var row2 = edge(tri.a, tri.b, p0) + tri.bias2;
  let dx0 = (tri.d.y - tri.b.y) * -16;
  let dy0 = (tri.d.x - tri.b.x) * 16;
  let dx1 = (tri.a.y - tri.d.y) * -16;
  let dy1 = (tri.a.x - tri.d.x) * 16;
  let dx2 = (tri.b.y - tri.a.y) * -16;
  let dy2 = (tri.b.x - tri.a.x) * 16;
  let page_size = i32(VSM_PAGE);
  for (var y = tri.start.y; y <= tri.hi.y; y++) {
    var w0 = row0;
    var w1 = row1;
    var w2 = row2;
    var x = tri.start.x;
    while (x <= tri.hi.x) {
      let at0 = tri.lo_abs + vec2i(x, y);
      let page = at0 >> vec2u(7u);
      let span = min(tri.hi.x - x + 1, page_size - (at0.x & (page_size - 1)));
      let phys = rendering(level, layer, page);
      if (phys == VSM_NONE) {
        w0 += dx0 * span;
        w1 += dx1 * span;
        w2 += dx2 * span;
        x += span;
        continue;
      }
      var drew = false;
      let e = x + span;
      for (; x < e; x++) {
        if ((w0 | w1 | w2) >= 0) {
          store(phys, layer, tri.lo_abs + vec2i(x, y), depth_at(tri, w0, w1, w2));
          drew = true;
        }
        w0 += dx0;
        w1 += dx1;
        w2 += dx2;
      }
      if (drew) { mark_moving(level, layer, page); }
    }
    row0 += dy0;
    row1 += dy1;
    row2 += dy2;
  }
}

// the whole workgroup: a page at a time, a texel an invocation.
fn walk_together(tri: Tri, level: u32, layer: u32, t: u32) {
  let page_size = i32(VSM_PAGE);
  let page_lo = (tri.lo_abs + tri.start) >> vec2u(7u);
  let page_hi = (tri.lo_abs + tri.hi) >> vec2u(7u);
  for (var py = page_lo.y; py <= page_hi.y; py++) {
    for (var px = page_lo.x; px <= page_hi.x; px++) {
      let phys = rendering(level, layer, vec2i(px, py));
      if (phys == VSM_NONE) { continue; }
      let lo = max(vec2i(px, py) * page_size - tri.lo_abs, tri.start);
      let hi = min(vec2i(px, py) * page_size + page_size - 1 - tri.lo_abs, tri.hi);
      let size = hi - lo + 1;
      var drew = false;
      for (var i = t; i < u32(size.x * size.y); i += 128u) {
        let q = lo + vec2i(i32(i) % size.x, i32(i) / size.x);
        let p = q * 16 + 8;
        let w0 = edge(tri.b, tri.d, p) + tri.bias0;
        let w1 = edge(tri.d, tri.a, p) + tri.bias1;
        let w2 = edge(tri.a, tri.b, p) + tri.bias2;
        if ((w0 | w1 | w2) < 0) { continue; }
        store(phys, layer, tri.lo_abs + q, depth_at(tri, w0, w1, w2));
        drew = true;
      }
      if (drew) { mark_moving(level, layer, vec2i(px, py)); }
    }
  }
}

@compute @workgroup_size(128)
fn vsm_raster(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) t: u32) {
  let k = wid.y * 65535u + wid.x;
  let valid = k < min(word(2), VSM_MAX_VISIBLE);
  var tagged = 0u;
  var cluster = 0u;
  if (valid) {
    tagged = word(VW_VISIBLE + 2u * k);
    cluster = word(VW_VISIBLE + 2u * k + 1u);
  }
  let level = (tagged >> 24u) & 15u;
  let layer = tagged >> 28u;
  let inst = load_instance(tagged & 0xffffffu);
  let c = clusters[cluster];
  let texel = vsm_texel(level);
  if (t == 0u) { atomicStore(&together_count, 0u); }
  if (valid && t < c.vertex_count) {
    let p = vsm_light_space(to_world(inst, cluster_position(c, meshes[inst.mesh].grid, t)));
    snapped[t] = vec2i(round(p.xy / texel * 16.0));
    depth[t] = p.z;
  }
  workgroupBarrier();
  if (valid && t < c.triangle_count) {
    let tri = setup(cluster_triangle(c, t), level, layer);
    if (tri.ok) {
      let size = tri.hi - tri.start + 1;
      if (size.x * size.y > 1024) {
        together[atomicAdd(&together_count, 1u)] = t;
      } else {
        walk(tri, level, layer);
      }
    }
  }
  workgroupBarrier();
  let count = atomicLoad(&together_count);
  for (var j = 0u; j < count; j++) {
    walk_together(setup(cluster_triangle(c, together[j]), level, layer), level, layer, t);
  }
}
