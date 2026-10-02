// The hardware path for large clusters: an ordinary render pipeline, since
// WebGPU has no mesh shaders. One instance per visible cluster, 384
// vertices each (128 triangles' worth); each vertex finds its cluster,
// triangle and corner and pulls its position from storage. Triangles past a
// cluster's count collapse to a point and draw nothing. The fragment writes
// (visible cluster, triangle) into an r32uint target, behind the depth test.

@group(1) @binding(0) var<storage, read> hw_visible: array<vec2u>;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) id: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOut {
  var out: VertexOut;
  let v = hw_visible[ii];
  let c = clusters[v.y];
  let tri = vi / 3u;
  out.id = (ii << 7u) | tri;
  if (tri >= c.triangle_count) {
    out.position = vec4f(2.0, 2.0, 2.0, 1.0);
    return out;
  }
  let packed = cluster_triangle(c, tri);
  let inst = instances[v.x];
  let p = cluster_position(c, meshes[inst.mesh].grid, (packed >> (8u * (vi % 3u))) & 255u);
  out.position = frame.view_proj * vec4f(to_world(inst, p), 1.0);
  return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) u32 {
  return in.id;
}

// Copies the shaded image to the canvas: a triangle covering the screen.
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

// The shadow map's casters: positions only, for depth.
@vertex
fn shadow_vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let v = hw_visible[ii];
  let c = clusters[v.y];
  let tri = vi / 3u;
  if (tri >= c.triangle_count) { return vec4f(2.0, 2.0, 2.0, 1.0); }
  let inst = instances[v.x];
  let packed = cluster_triangle(c, tri);
  let p = cluster_position(c, meshes[inst.mesh].grid, (packed >> (8u * (vi % 3u))) & 255u);
  return frame.view_proj * vec4f(to_world(inst, p), 1.0);
}
