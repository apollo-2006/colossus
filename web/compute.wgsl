// The web demo's compute passes: instance culling, cluster culling, the
// software rasterizer, and shading. A port of the Vulkan viewer's, shaped by
// what WebGPU lacks:
//
// * No mesh shaders: large clusters are drawn by raster.wgsl, an ordinary
//   render pipeline that pulls each cluster's triangles from storage.
// * No 64-bit atomics: the software rasterizer cannot write depth and
//   triangle in one atomic. It runs twice instead, first keeping the
//   nearest depth per pixel with a 32-bit atomic max, then writing the
//   triangle wherever its depth is the one that won.
// * No ray queries, so no shadows.
//
// The hardware and software results are kept apart and shade() takes the
// nearer of the two at each pixel.

struct Counters {
  work: atomic<u32>,
  hw: atomic<u32>,
  sw: atomic<u32>,
  instances_visible: atomic<u32>,
  clusters_tested: atomic<u32>,
  triangles: atomic<u32>,
  overflow: atomic<u32>,
  pad: u32,
}

@group(1) @binding(0) var<storage, read_write> counters: Counters;
@group(1) @binding(1) var<storage, read_write> work: array<vec2u>;        // (instance, first cluster)
@group(1) @binding(2) var<storage, read_write> hw_visible: array<vec2u>;  // (instance, cluster)
@group(1) @binding(3) var<storage, read_write> sw_visible: array<vec2u>;
@group(1) @binding(4) var<storage, read_write> sw_depth: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> sw_id: array<u32>;
@group(2) @binding(0) var hw_depth: texture_depth_2d;
@group(2) @binding(1) var hw_id: texture_2d<u32>;
@group(2) @binding(2) var out_image: texture_storage_2d<rgba8unorm, write>;
// The indirect arguments: [0, 3) the cluster culling dispatch, [4, 8) the
// hardware draw, [8, 11) the software rasterizer's dispatch. Bound only
// for the two passes that write them: a dispatch may not both read a
// buffer as its arguments and have it bound for writing.
@group(3) @binding(0) var<storage, read_write> args: array<u32>;

// --- Instance culling: one workgroup per instance. One invocation decides
// and finds where the drawable clusters begin; all 64 write the work items.
// Clusters are stored by parent error, and one can only be drawn while its
// parent looks too coarse, so every cluster whose parent error is under
// threshold * distance / (scale * lod_scale), distance measured to the
// model's LOD sphere, is skipped with a binary search.

var<workgroup> wg_first_item: u32;
var<workgroup> wg_items: u32;
var<workgroup> wg_first_cluster: u32;

@compute @workgroup_size(64)
fn instance_cull(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
  let i = wid.y * 65535u + wid.x;
  if (i >= frame.instance_count) { return; }
  if (lane == 0u) {
    wg_items = 0u;
    let inst = instances[i];
    let m = meshes[inst.mesh];
    let center = to_world(inst, m.bounds.xyz);
    if ((frame.flags & FLAG_FRUSTUM) == 0u || sphere_in_frustum(center, m.bounds.w * inst.scale)) {
      atomicAdd(&counters.instances_visible, 1u);
      let lc = to_world(inst, m.lod_bounds.xyz);
      let d = max(length(lc - frame.cull_origin.xyz) - m.lod_bounds.w * inst.scale, frame.near_z);
      let limit = frame.lod_threshold * d / (inst.scale * frame.lod_scale);
      var lo = 0u;
      var hi = m.cluster_count;
      while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (clusters[m.first_cluster + mid].parent_error > limit) { hi = mid; } else { lo = mid + 1u; }
      }
      var n = (m.cluster_count - lo + 63u) / 64u;
      var base = 0u;
      if (n > 0u) { base = atomicAdd(&counters.work, n); }
      if (base + n > frame.max_work) {
        atomicAdd(&counters.overflow, 1u);
        n = select(0u, frame.max_work - base, base < frame.max_work);
      }
      wg_first_item = base;
      wg_items = n;
      wg_first_cluster = m.first_cluster + lo;
    }
  }
  let n = workgroupUniformLoad(&wg_items);
  for (var k = lane; k < n; k += 64u) {
    work[wg_first_item + k] = vec2u(i, wg_first_cluster + 64u * k);
  }
}

fn rows(n: u32, at: u32) {
  args[at] = min(n, 65535u);
  args[at + 1u] = (n + 65534u) / 65535u;
  args[at + 2u] = 1u;
}

@compute @workgroup_size(1)
fn args_cull() {
  rows(min(atomicLoad(&counters.work), frame.max_work), 0u);
}

// --- Cluster culling: one workgroup per work item, one invocation per
// cluster: the LOD cut, the frustum and the normal cone. Survivors small on
// screen go to the software rasterizer, the rest to the hardware.

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
  let item = wid.y * 65535u + wid.x;
  var draw = false;
  var software = false;
  var instance_id = 0u;
  var cluster_id = 0u;
  if (item < min(atomicLoad(&counters.work), frame.max_work)) {
    let w = work[item];
    let inst = instances[w.x];
    let m = meshes[inst.mesh];
    instance_id = w.x;
    cluster_id = w.y + lane;
    if (cluster_id < m.first_cluster + m.cluster_count) {
      atomicAdd(&wg_tested, 1u);
      let c = clusters[cluster_id];
      let s = inst.scale;
      let tau = frame.lod_threshold;
      draw = projected_error(to_world(inst, c.lod_center), c.lod_radius * s, c.lod_error * s) <= tau &&
             projected_error(to_world(inst, c.parent_center), c.parent_radius * s, c.parent_error * s) > tau;
      let center = to_world(inst, c.center);
      let r = c.radius * s;
      if (draw && (frame.flags & FLAG_FRUSTUM) != 0u) { draw = sphere_in_frustum(center, r); }
      if (draw && (frame.flags & FLAG_CONE) != 0u && c.cone_cutoff < 1.0) {
        let axis = normalize(to_world_dir(inst, c.cone_axis));
        let view = center - frame.cull_origin.xyz;
        if (dot(view, axis) >= c.cone_cutoff * length(view) + r) { draw = false; }
      }
      if (draw) {
        atomicAdd(&wg_triangles, c.triangle_count);
        // The sphere's size on screen, roughly; clusters reaching the
        // near plane always go to the hardware, which clips.
        let d = length(center - frame.origin.xyz) - r;
        software = (frame.flags & FLAG_SOFTWARE) != 0u && d > frame.near_z * 2.0 &&
                   2.0 * r * frame.lod_scale / d < frame.sw_max_pixels;
      }
    }
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

@compute @workgroup_size(1)
fn args_draw() {
  // drawIndirect: 128 triangles' worth of vertices per cluster, one
  // instance per cluster.
  args[4] = 384u;
  args[5] = min(atomicLoad(&counters.hw), frame.max_visible);
  args[6] = 0u;
  args[7] = 0u;
  rows(min(atomicLoad(&counters.sw), frame.max_visible), 8u);
}

// --- The software rasterizer: one workgroup per small cluster, one
// invocation per vertex and then per triangle. Vertices are snapped to
// 1/256 pixel, edges follow a top-left rule, and edge functions are
// measured from each triangle's corner so they fit 32 bits (clusters sent
// here are small). Pass 1 keeps the nearest depth per pixel; pass 2 writes
// the triangle wherever its depth won.

var<workgroup> screen: array<vec2i, 128>;
var<workgroup> depth: array<f32, 128>;
var<workgroup> wg_valid: u32;

fn edge(a: vec2i, b: vec2i, p: vec2i) -> i32 {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Top-left rule, y down: an edge running up (or flat and running right, in
// this winding) owns the pixel centers on it.
fn owns_edge(a: vec2i, b: vec2i) -> bool {
  let d = b - a;
  return d.y < 0 || (d.y == 0 && d.x > 0);
}

fn sw_raster(wid: vec3u, lane: u32, write_id: bool) {
  let k = wid.y * 65535u + wid.x;
  if (lane == 0u) { wg_valid = select(0u, 1u, k < min(atomicLoad(&counters.sw), frame.max_visible)); }
  if (workgroupUniformLoad(&wg_valid) == 0u) { return; }
  let v = sw_visible[k];
  let inst = instances[v.x];
  let c = clusters[v.y];
  if (lane < c.vertex_count) {
    let clip = frame.view_proj * vec4f(to_world(inst, cluster_position(c, lane)), 1.0);
    let ndc = clip.xy / clip.w;
    // Pixels run down; clip space runs up.
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
  let lo_abs = max((min(a, min(b, d)) - vec2i(128) + vec2i(255)) >> vec2u(8u), vec2i(0));
  let hi = min((max(a, max(b, d)) - vec2i(128)) >> vec2u(8u), vec2i(i32(frame.width), i32(frame.height)) - vec2i(1)) - lo_abs;
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
  // Both sides are drawn: put every triangle in one winding.
  if (area < 0) {
    let t = b; b = d; d = t;
    let tz = zb; zb = zd; zd = tz;
    area = -area;
  }
  let bias0 = select(-1, 0, owns_edge(b, d));
  let bias1 = select(-1, 0, owns_edge(d, a));
  let bias2 = select(-1, 0, owns_edge(a, b));
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
          if (atomicLoad(&sw_depth[at]) == bits) { sw_id[at] = id; }
        } else if (bits > atomicLoad(&sw_depth[at])) {
          atomicMax(&sw_depth[at], bits);
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

// --- Shading: the nearer of the hardware and software results, its
// triangle fetched again and intersected with the pixel's ray.

const SUN_DIR = normalize(vec3f(0.75, 0.5, 0.3));
const SUN_COLOR = vec3f(1.0, 0.92, 0.82) * 1.7;

fn sky(dir: vec3f) -> vec3f {
  let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  return mix(vec3f(0.66, 0.68, 0.71), vec3f(0.3, 0.42, 0.62), pow(t, 0.7)) * 0.9;
}

fn hash_color(x: u32) -> vec3f {
  let h = hash(x);
  return vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0 * 0.8 + 0.1;
}

fn level_color(level: u32) -> vec3f {
  let h = 0.66 - 0.055 * f32(min(level, 16u));
  let k = abs(fract(vec3f(h) + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return mix(vec3f(1.0), clamp(k - 1.0, vec3f(0.0), vec3f(1.0)), 0.75) * 0.85;
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn srgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
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
  let sd_bits = atomicLoad(&sw_depth[gid.x + gid.y * frame.width]);
  let sd = bitcast<f32>(sd_bits);
  var found = false;
  var software = false;
  var z = 0.0;  // The depth drawn, reversed
  var vc = vec2u(0u);
  var tri = 0u;
  var id = 0u;
  if (hd > 0.0 && hd >= sd) {
    id = textureLoad(hw_id, px, 0).x;
    vc = hw_visible[id >> 7u];
    found = true;
    z = hd;
  } else if (sd_bits != 0u) {
    id = sw_id[gid.x + gid.y * frame.width];
    vc = sw_visible[id >> 7u];
    found = true;
    software = true;
    z = sd;
  }
  tri = id & 127u;

  var color = sky(dir);
  if (!found && frame.debug_mode == 7u) {
    color = vec3f(1.0, 0.0, 1.0);  // Holes: anything not drawn
  } else if (!found) {
    if (dir.y < 0.0) {
      let t = -origin.y / dir.y;
      let hit = origin + dir * t;
      let cell = abs(fract(hit.xz * 0.5) - 0.5);
      let line = smoothstep(0.47, 0.5, max(cell.x, cell.y));
      let ground = mix(vec3f(0.42, 0.4, 0.37), vec3f(0.33, 0.31, 0.29), line) *
                   (SUN_COLOR * SUN_DIR.y * 0.6 + sky(vec3f(0.0, 1.0, 0.0)) * 0.45);
      color = mix(ground, color, 1.0 - exp(-t * 0.012));
    }
  } else {
    let inst = instances[vc.x];
    let c = clusters[vc.y];
    let packed = cluster_triangle(c, tri);
    let i0 = packed & 255u;
    let i1 = (packed >> 8u) & 255u;
    let i2 = (packed >> 16u) & 255u;
    let p0 = to_world(inst, cluster_position(c, i0));
    let p1 = to_world(inst, cluster_position(c, i1));
    let p2 = to_world(inst, cluster_position(c, i2));
    // The ray against the triangle's plane, for exact barycentrics.
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let pv = cross(dir, e2);
    let det = dot(e1, pv);
    let inv = select(0.0, 1.0 / det, abs(det) > 1e-20);
    let tv = origin - p0;
    let qv = cross(tv, e1);
    // Clamped: at a silhouette the triangle under a pixel center can be
    // nearly edge-on, and the ray meets its plane far outside it.
    let bu = clamp(dot(tv, pv) * inv, 0.0, 1.0);
    let bv = clamp(dot(dir, qv) * inv, 0.0, 1.0 - bu);
    // The distance from depth, which is exact, not from that plane.
    let center = frame.inv_view_proj * vec4f(0.0, 0.0, 1.0, 1.0);
    let forward = normalize(center.xyz / center.w - origin);
    let t = frame.near_z / z / dot(dir, forward);
    let n0 = cluster_normal(c, i0);
    let n1 = cluster_normal(c, i1);
    let n2 = cluster_normal(c, i2);
    var n = normalize(to_world_dir(inst, n0 * (1.0 - bu - bv) + n1 * bu + n2 * bv));
    // Seen from behind (the inside of a fold, or through a hole in the
    // scan): light the side we see, and leave the rim light off, which
    // would outline every such sliver.
    let behind = dot(n, dir) > 0.0;
    if (behind) { n = -n; }

    var albedo = vec3f(0.56, 0.52, 0.47);
    switch (frame.debug_mode) {
      case 1u: { albedo = hash_color(vc.y * 7919u + vc.x * 104729u); }
      case 2u: { albedo = hash_color(id * 2654435761u + vc.x + select(0u, 0x9e3779b9u, software)); }
      case 3u: { albedo = level_color(c.level); }
      case 4u: { albedo = hash_color(c.group + vc.x * 104729u); }
      case 5u: { albedo = hash_color(vc.x); }
      case 6u: { albedo = select(vec3f(0.15, 0.45, 0.95), vec3f(0.95, 0.45, 0.1), software); }
      case 7u: { albedo = vec3f(0.3); }
      default: {}
    }
    let diffuse = max(dot(n, SUN_DIR), 0.0);
    let ambient = mix(vec3f(0.24, 0.22, 0.2), sky(vec3f(0.0, 1.0, 0.0)), n.y * 0.5 + 0.5) * 0.42;
    let rim = select(pow(1.0 - max(dot(n, -dir), 0.0), 4.0) * 0.25, 0.0, behind);
    color = albedo * (SUN_COLOR * diffuse + ambient) + rim * sky(n);
    color = mix(color, sky(dir), 1.0 - exp(-t * 0.012));
  }
  textureStore(out_image, px, vec4f(srgb(aces(color)), 1.0));
}
