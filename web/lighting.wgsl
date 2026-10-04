// light from everything but the sun, as the viewer's lighting.glsl: shared by shading
// (compute.wgsl) and the indirect pass (ao.wgsl).

const SUN_DIR = normalize(vec3f(0.75, 0.5, 0.3));
const SUN_COLOR = vec3f(1.0, 0.92, 0.82) * 1.7;
const FLAG_BOUNCE = 2048u;  // light bounced off the ground and nearby surfaces

fn sky(dir: vec3f) -> vec3f {
  let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  return mix(vec3f(0.66, 0.68, 0.71), vec3f(0.3, 0.42, 0.62), pow(t, 0.7)) * 0.9;
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

// aces() undone (its quadratic solved for x).
fn aces_inverse(y0: vec3f) -> vec3f {
  let y = clamp(y0, vec3f(0.0), vec3f(0.99));
  let a = 2.43 * y - 2.51;
  let b = 0.59 * y - 0.03;
  let c = 0.14 * y;
  return (-b - sqrt(max(b * b - 4.0 * a * c, vec3f(0.0)))) / (2.0 * a);
}

fn srgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
}

fn srgb_inverse(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

// the ground's colour and the light on it, as drawn, in sun (lit 1) or shadow (lit 0).
const GROUND_ALBEDO = vec3f(0.41, 0.39, 0.36);

fn ground_radiance(lit: f32, ao: f32) -> vec3f {
  return GROUND_ALBEDO * (SUN_COLOR * max(SUN_DIR.y, 0.0) * 0.6 * lit + sky(vec3f(0.0, 1.0, 0.0)) * 0.45 * ao);
}

// the old ambient: sky above, a fixed grey below.
fn plain_ambient(n: vec3f, ao: f32) -> vec3f {
  return mix(vec3f(0.24, 0.22, 0.2), sky(vec3f(0.0, 1.0, 0.0)), n.y * 0.5 + 0.5) * 0.42 * ao;
}
