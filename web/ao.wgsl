// ground-truth ambient occlusion, as the viewer's ao.comp and ao_depth.comp: a half-resolution
// view depth chain, then two slices per pixel rotated per frame for taa to average.

@group(1) @binding(0) var hw_depth: texture_depth_2d;
@group(1) @binding(1) var<storage, read> sw_depth: array<u32>;
@group(1) @binding(2) var depth_out: texture_storage_2d<r32float, write>;
@group(1) @binding(3) var depth_in: texture_2d<f32>;        // the level below
@group(1) @binding(4) var depth_chain: texture_2d<f32>;     // all levels
@group(1) @binding(5) var ao_out: texture_storage_2d<r32float, write>;

const AO_RADIUS = 0.25;  // world units
const AO_SLICES = 2u;
const AO_STEPS = 5u;     // per side
const AO_POWER = 2.2;    // xegtao's final power
const PI = 3.14159265;

// view ray of a full-resolution pixel centre, at w = 1. webgpu's ndc y runs up.
fn ray(px: vec2f) -> vec3f {
  let ndc = vec2f(px.x / f32(frame.width) * 2.0 - 1.0, 1.0 - px.y / f32(frame.height) * 2.0);
  return vec3f(ndc.x / frame.p00, ndc.y / frame.p11, -1.0);
}

// level 0: view depth (w) at the top-left pixel of each 2x2; ground or far where nothing drew.
@compute @workgroup_size(8, 8)
fn ao_depth_first(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(depth_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let px = min(gid.xy * 2u, vec2u(frame.width, frame.height) - 1u);
  let z = max(textureLoad(hw_depth, px, 0), bitcast<f32>(sw_depth[px.x + px.y * frame.width]));
  var w = 1e5;
  if (z > 0.0) {
    w = frame.near_z / z;
  } else {
    // ground, y = 0: world y of a view point is dot(view[1].xyz, p) + eye y.
    let down = dot(frame.view[1].xyz, ray(vec2f(px) + 0.5));
    if (down < 0.0) { w = -frame.origin.y / down; }
  }
  textureStore(depth_out, gid.xy, vec4f(w));
}

// later levels: each texel its block's top-left.
@compute @workgroup_size(8, 8)
fn ao_depth_down(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(depth_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  textureStore(depth_out, gid.xy, textureLoad(depth_in, gid.xy * 2u, 0));
}

fn view_position(q_in: vec2i, level: i32) -> vec3f {
  let size = vec2i(textureDimensions(depth_chain, level));
  let q = clamp(q_in, vec2i(0), size - 1);
  return ray(vec2f(q << vec2u(u32(level + 1))) + 0.5) * textureLoad(depth_chain, q, level).x;
}

// interleaved gradient noise (jimenez 2014).
fn noise(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8)
fn ao(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(ao_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let q = vec2i(gid.xy);
  let p = view_position(q, 0);
  if (-p.z > 1e4) {
    textureStore(ao_out, gid.xy, vec4f(1.0));
    return;
  }
  // normal from depth: the smaller difference either side.
  let l = view_position(q - vec2i(1, 0), 0);
  let r = view_position(q + vec2i(1, 0), 0);
  let u = view_position(q - vec2i(0, 1), 0);
  let d = view_position(q + vec2i(0, 1), 0);
  let dx = select(p - l, r - p, abs(r.z - p.z) < abs(p.z - l.z));
  let dy = select(p - u, d - p, abs(d.z - p.z) < abs(p.z - u.z));
  var n = normalize(cross(dy, dx));
  let v = normalize(-p);
  if (dot(n, v) < 0.0) { n = -n; }
  let radius_px = min(AO_RADIUS * frame.lod_scale / max(-p.z, frame.near_z), 96.0);
  if (radius_px < 1.0) {
    textureStore(ao_out, gid.xy, vec4f(1.0));
    return;
  }
  let falloff_range = 0.615 * AO_RADIUS;
  let falloff_mul = -1.0 / falloff_range;
  let falloff_add = (AO_RADIUS - falloff_range) / falloff_range + 1.0;
  let frame_shift = f32(frame.frame_index % 64u);
  let jitter = vec2f(noise(vec2f(q) + frame_shift * 5.588238), noise(vec2f(q.yx) + frame_shift * 7.123));

  var visibility = 0.0;
  for (var slice = 0u; slice < AO_SLICES; slice++) {
    let phi = (f32(slice) + jitter.x) * PI / f32(AO_SLICES);
    let omega = vec2f(cos(phi), sin(phi));  // screen, y down
    let direction = vec3f(omega.x, -omega.y, 0.0);  // view, y up
    let ortho = direction - dot(direction, v) * v;
    let axis = normalize(cross(ortho, v));
    let projected = n - axis * dot(n, axis);
    let projected_length = length(projected);
    let cos_n = clamp(dot(projected, v) / max(projected_length, 1e-6), 0.0, 1.0);
    let angle_n = sign(dot(ortho, projected)) * acos(cos_n);
    let low0 = cos(angle_n + PI * 0.5);
    let low1 = cos(angle_n - PI * 0.5);
    var horizon0 = low0;
    var horizon1 = low1;
    for (var step = 0u; step < AO_STEPS; step++) {
      // quadratic spacing in half-resolution texels; far taps on coarser levels.
      var s = (f32(step) + jitter.y) / f32(AO_STEPS);
      s = max(s * s * radius_px * 0.5, 1.0);
      let level = clamp(i32(log2(s)) - 1, 0, 3);
      let o = vec2i(round(omega * s));
      let h0 = view_position((q + o) >> vec2u(u32(level)), level) - p;
      let h1 = view_position((q - o) >> vec2u(u32(level)), level) - p;
      let d0 = length(h0);
      let d1 = length(h1);
      let w0 = clamp(d0 * falloff_mul + falloff_add, 0.0, 1.0);
      let w1 = clamp(d1 * falloff_mul + falloff_add, 0.0, 1.0);
      horizon0 = max(horizon0, mix(low0, dot(h0 / max(d0, 1e-6), v), w0));
      horizon1 = max(horizon1, mix(low1, dot(h1 / max(d1, 1e-6), v), w1));
    }
    var a0 = -acos(clamp(horizon1, -1.0, 1.0));
    var a1 = acos(clamp(horizon0, -1.0, 1.0));
    a0 = angle_n + clamp(a0 - angle_n, -PI * 0.5, PI * 0.5);
    a1 = angle_n + clamp(a1 - angle_n, -PI * 0.5, PI * 0.5);
    let arc0 = (cos_n + 2.0 * a0 * sin(angle_n) - cos(2.0 * a0 - angle_n)) * 0.25;
    let arc1 = (cos_n + 2.0 * a1 * sin(angle_n) - cos(2.0 * a1 - angle_n)) * 0.25;
    visibility += projected_length * (arc0 + arc1);
  }
  textureStore(ao_out, gid.xy, vec4f(pow(clamp(visibility / f32(AO_SLICES), 0.0, 1.0), AO_POWER)));
}
