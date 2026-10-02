// hardware path for big clusters, a render pipeline for want of mesh shaders:
// an instance per visible cluster, 384 vertices each; each vertex finds its
// cluster, triangle and corner and pulls its position. triangles past the count
// collapse to a point. the fragment writes (visible cluster, triangle) to an
// r32uint target behind the depth test.

@group(1) @binding(0) var<storage, read> hw_visible: array<vec2u>;
struct PassInfo {
  pass_index: u32,
  level: u32,
  pad0: u32,
  pad1: u32,
}
@group(1) @binding(2) var<uniform> pass_info: PassInfo;
// compute.wgsl's Counters, for each pass's start in hw_visible (written by
// args_draw).
struct Counters {
  counts: array<u32, 12>,
  pass_start: array<u32, 8>,
}
@group(1) @binding(3) var<storage, read> counters: Counters;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) id: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) local: u32) -> VertexOut {
  var out: VertexOut;
  let ii = counters.pass_start[pass_info.pass_index] + local;
  let v = hw_visible[ii];
  let c = clusters[v.y];
  let tri = vi / 3u;
  out.id = (ii << 7u) | tri;
  if (tri >= c.triangle_count) {
    out.position = vec4f(2.0, 2.0, 2.0, 1.0);
    return out;
  }
  let packed = cluster_triangle(c, tri);
  let inst = load_instance(v.x);
  let p = cluster_position(c, meshes[inst.mesh].grid, (packed >> (8u * (vi % 3u))) & 255u);
  out.position = frame.view_proj * vec4f(to_world(inst, p), 1.0);
  return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) u32 {
  return in.id;
}

// shaded image to the canvas: one screen-covering triangle.
@group(1) @binding(1) var shaded: texture_2d<f32>;

@vertex
fn blit_vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn blit_fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return textureLoad(shaded, vec2i(p.xy), 0);
}
