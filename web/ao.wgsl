// ground-truth ambient occlusion, as the viewer's ao.comp and ao_depth.comp: a half-resolution
// view depth chain, then two slices per pixel rotated per frame for taa to average.

@group(1) @binding(0) var hw_depth: texture_depth_2d;
@group(1) @binding(1) var<storage, read> sw_depth: array<u32>;
@group(1) @binding(2) var depth_out: texture_storage_2d<r32float, write>;
@group(1) @binding(3) var depth_in: texture_2d<f32>;        // the level below
@group(1) @binding(4) var depth_chain: texture_2d<f32>;     // all levels
@group(1) @binding(5) var ao_out: texture_storage_2d<rgba16float, write>;  // indirect light, occlusion
// for the ground's bounce: the shadow pages; for nearby bounce: last frame's image (taa's history).
@group(1) @binding(6) var<storage, read> vsm_entries: array<u32>;
@group(1) @binding(7) var<storage, read> vsm_atlas: array<u32>;
@group(1) @binding(8) var history: texture_2d<f32>;
@group(1) @binding(9) var history_sampler: sampler;

const AO_RADIUS = 0.25;  // world units
const AO_SLICES = 2u;
const AO_STEPS = 5u;     // per side
const AO_POWER = 2.2;    // xegtao's final power
const PI = 3.14159265;

// the inverse of frame.view: a rotation and a translation.
fn inverse_view() -> mat4x4f {
  let r = transpose(mat3x3f(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz));
  let t = -(r * frame.view[3].xyz);
  return mat4x4f(vec4f(r[0], 0.0), vec4f(r[1], 0.0), vec4f(r[2], 0.0), vec4f(t, 1.0));
}

// the visible cosine weight from the normal to a horizon at angle a (gtao's integral).
fn arc(cos_n: f32, angle_n: f32, a0: f32) -> f32 {
  let a = angle_n + clamp(a0 - angle_n, -PI * 0.5, PI * 0.5);
  return (cos_n + 2.0 * a * sin(angle_n) - cos(2.0 * a - angle_n)) * 0.25;
}

// the light a view-space point showed last frame: reprojected into the history, srgb and tone
// curve undone. black off screen.
fn last_light(to_world_space: mat4x4f, view_p: vec3f) -> vec3f {
  let clip = frame.prev_view_proj * (to_world_space * vec4f(view_p, 1.0));
  if (clip.w <= 0.0) { return vec3f(0.0); }
  let ndc = clip.xy / clip.w;
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);  // webgpu textures run down
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return vec3f(0.0); }
  return aces_inverse(srgb_inverse(textureSampleLevel(history, history_sampler, uv, 0.0).rgb));
}

// sunlight at a ground point, hard edged, from the shadow pages (the plain level, then up).
fn ground_lit(g: vec3f) -> f32 {
  for (var level = vsm_level_for(length(g - frame.origin.xyz)); level < VSM_LEVELS; level++) {
    let texel = vsm_texel(level);
    let lp = vsm_light_space(g + vec3f(0.0, 2.0 * texel, 0.0));
    let at = vec2i(floor(lp.xy / texel));
    let page = at >> vec2u(7u);
    if (!vsm_in_window(level, page)) { continue; }
    let slot = vsm_slot(level, page);
    if (vsm_entries[4u * slot] != vsm_tag(page) || vsm_entries[4u * slot + 1u] == VSM_NONE) { continue; }
    let phys = vsm_entries[4u * slot + 1u];
    let in_page = vec2u(at & vec2i(i32(VSM_PAGE) - 1));
    var stored = vsm_atlas[vsm_atlas_index(phys, in_page, VSM_STILL)];
    if ((vsm_entries[4u * slot + 3u] & VSM_HAS_MOVING) != 0u) { stored = max(stored, vsm_atlas[vsm_atlas_index(phys, in_page, VSM_MOVING)]); }
    return select(1.0, 0.0, stored != 0u && vsm_unsortable(stored) > lp.z + 1.5 * texel);
  }
  return 1.0;
}

// light arriving at p (normal n) from all but the sun, times occlusion, as lighting.glsl's
// ambient_light(): the sky above, and four rays down onto the ground, each taking the ground's
// light where it lands, in sun or in shadow.
fn ambient_light(p: vec3f, n: vec3f, ao: f32, noise_value: f32) -> vec3f {
  let up_light = sky(vec3f(0.0, 1.0, 0.0)) * 0.42;
  if ((frame.flags & (FLAG_BOUNCE | FLAG_SHADOWS)) != (FLAG_BOUNCE | FLAG_SHADOWS)) { return plain_ambient(n, ao); }
  let t = normalize(select(cross(n, vec3f(1.0, 0.0, 0.0)), cross(n, vec3f(0.0, 1.0, 0.0)), abs(n.y) < 0.9));
  let b = cross(n, t);
  var sum = vec3f(0.0);
  for (var k = 0u; k < 4u; k++) {
    let r = sqrt((f32(k) + 0.5) / 4.0);
    let a = f32(k) * 2.39996 + noise_value * 6.2831853;
    let dir = normalize(t * (r * cos(a)) + b * (r * sin(a)) + n * sqrt(max(1.0 - r * r, 0.0)));
    if (dir.y >= 0.0 || p.y <= 0.0) {
      sum += up_light;
      continue;
    }
    sum += ground_radiance(ground_lit(p + dir * (-p.y / dir.y)), 1.0);
  }
  return sum * 0.25 * ao;
}

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
    textureStore(ao_out, gid.xy, vec4f(0.0, 0.0, 0.0, 1.0));
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
  // world space, for the ground's bounce and reading last frame.
  let to_world_space = inverse_view();
  let world_p = (to_world_space * vec4f(p, 1.0)).xyz;
  let world_n = normalize((to_world_space * vec4f(n, 0.0)).xyz);
  let bounce = (frame.flags & FLAG_BOUNCE) != 0u && frame.taa_valid != 0u;
  var bounced = vec3f(0.0);
  let sky_noise = noise(vec2f(q) + f32(frame.frame_index % 64u) * 3.17);

  let radius_px = min(AO_RADIUS * frame.lod_scale / max(-p.z, frame.near_z), 96.0);
  if (radius_px < 1.0) {
    textureStore(ao_out, gid.xy, vec4f(ambient_light(world_p, world_n, 1.0, sky_noise), 1.0));
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
      let c0 = mix(low0, dot(h0 / max(d0, 1e-6), v), w0);
      let c1 = mix(low1, dot(h1 / max(d1, 1e-6), v), w1);
      if (bounce && c0 > horizon0) {
        let hidden = abs(arc(cos_n, angle_n, acos(clamp(horizon0, -1.0, 1.0))) - arc(cos_n, angle_n, acos(clamp(c0, -1.0, 1.0))));
        bounced += projected_length * hidden * last_light(to_world_space, p + h0);
      }
      if (bounce && c1 > horizon1) {
        let hidden = abs(arc(cos_n, angle_n, -acos(clamp(horizon1, -1.0, 1.0))) - arc(cos_n, angle_n, -acos(clamp(c1, -1.0, 1.0))));
        bounced += projected_length * hidden * last_light(to_world_space, p + h1);
      }
      horizon0 = max(horizon0, c0);
      horizon1 = max(horizon1, c1);
    }
    var a0 = -acos(clamp(horizon1, -1.0, 1.0));
    var a1 = acos(clamp(horizon0, -1.0, 1.0));
    a0 = angle_n + clamp(a0 - angle_n, -PI * 0.5, PI * 0.5);
    a1 = angle_n + clamp(a1 - angle_n, -PI * 0.5, PI * 0.5);
    let arc0 = (cos_n + 2.0 * a0 * sin(angle_n) - cos(2.0 * a0 - angle_n)) * 0.25;
    let arc1 = (cos_n + 2.0 * a1 * sin(angle_n) - cos(2.0 * a1 - angle_n)) * 0.25;
    visibility += projected_length * (arc0 + arc1);
  }
  let ao = pow(clamp(visibility / f32(AO_SLICES), 0.0, 1.0), AO_POWER);
  // the bounce at the scale ambient light is drawn at (the sky's 0.42), as ao.comp.
  let indirect = ambient_light(world_p, world_n, ao, sky_noise) + bounced * (0.42 / f32(AO_SLICES));
  textureStore(ao_out, gid.xy, vec4f(indirect, ao));
}
