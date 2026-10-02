// What is under a pixel: shared by shade.comp and shadow.comp, which must
// agree on it exactly. The visible triangle is fetched again, the camera
// ray through the pixel center meets its plane for barycentrics, and the
// distance comes from the depth drawn. Pixels with nothing drawn are the
// ground, or the sky.


const uint surface_sky = 0u, surface_ground = 1u, surface_object = 2u;

struct Surface {
    uint kind;
    vec3 dir;     // From the camera, through the pixel
    float t;      // Distance along dir
    vec3 hit;
    vec3 n;       // Facing the camera
    bool behind;  // The back of the surface: the inside of a fold, or through a hole
    float bias;   // How far a shadow ray must start above it, for the drawn surface's own error
    uint id;      // In the visibility buffer
    uvec2 vc;     // (instance, cluster)
    vec3 p0, p1, p2;
};

vec3 pixel_ray(uvec2 px) {
    const vec2 ndc = (vec2(px) + 0.5) / vec2(frame.width, frame.height) * 2.0 - 1.0;
    const vec4 near_point = frame.inv_view_proj * vec4(ndc, 1.0, 1.0);
    return normalize(near_point.xyz / near_point.w - frame.origin.xyz);
}

Surface surface_at(uvec2 px) {
    Surface s;
    s.dir = pixel_ray(px);
    s.behind = false;
    s.bias = 0.0;
    s.id = 0u;
    s.vc = uvec2(0u);
    const vec3 origin = frame.origin.xyz;
    const uint64_t v = vis[px.x + px.y * frame.width];
    if (v == 0ul) {
        s.kind = s.dir.y < 0.0 ? surface_ground : surface_sky;
        s.t = s.dir.y < 0.0 ? -origin.y / s.dir.y : 1e30;
        s.hit = origin + s.dir * s.t;
        s.n = vec3(0, 1, 0);
        s.bias = 1e-4 + s.t * 1e-5;
        return s;
    }
    s.kind = surface_object;
    s.id = uint(v & 0xffffffffu);
    s.vc = visible[s.id >> 7];
    const Instance inst = load_instance(s.vc.x);
    const Cluster c = clusters[s.vc.y];
    const uint base = page_table[c.group];
    const uint packed = cluster_triangle(base, c, s.id & 127u);
    const uint i0 = packed & 255u, i1 = (packed >> 8) & 255u, i2 = (packed >> 16) & 255u;
    const vec4 grid = meshes[inst.mesh].grid;
    s.p0 = to_world(inst, cluster_position(base, c, grid, i0));
    s.p1 = to_world(inst, cluster_position(base, c, grid, i1));
    s.p2 = to_world(inst, cluster_position(base, c, grid, i2));

    // The ray against the triangle's plane (Moller-Trumbore, without the
    // bounds checks: the rasterizer already said it hits).
    const vec3 e1 = s.p1 - s.p0, e2 = s.p2 - s.p0;
    const vec3 pv = cross(s.dir, e2);
    const float det = dot(e1, pv);
    const vec3 tv = origin - s.p0;
    const float inv = abs(det) > 1e-20 ? 1.0 / det : 0.0;
    const vec3 qv = cross(tv, e1);
    // Clamped: at a silhouette the triangle under a pixel center can be
    // nearly edge-on, and the ray meets its plane far outside it.
    const float bu = clamp(dot(tv, pv) * inv, 0.0, 1.0);
    const float bv = clamp(dot(s.dir, qv) * inv, 0.0, 1.0 - bu);
    // The distance from depth, which is exact, not from that plane.
    const vec4 center = frame.inv_view_proj * vec4(0.0, 0.0, 1.0, 1.0);
    const vec3 forward = normalize(center.xyz / center.w - origin);
    s.t = frame.near_z / uintBitsToFloat(uint(v >> 32)) / dot(s.dir, forward);
    s.hit = origin + s.dir * s.t;
    s.n = normalize(to_world_dir(inst, cluster_normal(base, c, i0) * (1.0 - bu - bv) + cluster_normal(base, c, i1) * bu +
                                           cluster_normal(base, c, i2) * bv));
    s.behind = dot(s.n, s.dir) > 0.0;
    if (s.behind) s.n = -s.n;
    // The drawn cut strays up to this cluster's error from the original;
    // trace_surface() adds the shadow copy's.
    s.bias = c.lod_error * inst.scale * 1.5 + 1e-4 + s.t * 2e-5;
    return s;
}

#ifdef RAY_QUERY
#extension GL_EXT_ray_query : require
layout(set = 0, binding = 18) uniform accelerationStructureEXT shadow_scene;   // Still instances
layout(set = 0, binding = 24) uniform accelerationStructureEXT shadow_moving;  // Moving ones, refitted every frame

bool hits(accelerationStructureEXT scene, vec3 start, uint mask) {
    rayQueryEXT q;
    rayQueryInitializeEXT(q, scene,
                          gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT, mask,
                          start, 0.0, sun_dir, (frame.scene_top - start.y) / sun_dir.y);
    while (rayQueryProceedEXT(q)) {
    }
    return rayQueryGetIntersectionTypeEXT(q, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

// 1 where the sun reaches p, 0 where something is in the way. The ray
// starts `bias` above the surface: the shadow geometry is a coarser copy
// and can stand that far above the surface drawn. Back faces are skipped,
// so where the drawn surface dips inside the copy, the ray leaves through
// the copy's back and is not stopped (the builder winds every model
// counterclockwise from outside).
//
// Nothing stands above frame.scene_top, so the ray stops where it climbs
// past it: that loses no shadow, and spares the traversal every instance
// the ray would otherwise cross on its way out of the crowd.
float trace_sun(vec3 p, vec3 n, float bias, uint mask) {
    const vec3 start = p + n * bias;
    if (start.y >= frame.scene_top) return 1.0;
    if (hits(shadow_scene, start, mask)) return 0.0;
    return (frame.flags & flag_moving) != 0u && hits(shadow_moving, start, mask) ? 0.0 : 1.0;
}

// Whether a surface faces the sun and is lit by it. The ray is traced
// against the coarsest shadow copy whose error, seen from the camera at
// this distance, is under half a pixel (occluders stand within a statue's
// shadow of what they shade, about as far away), and clears that error.
float trace_surface(Surface s) {
    if (s.kind == surface_sky) return 1.0;
    if (s.kind == surface_ground && s.t > 200.0) return 1.0;
    if (dot(s.n, sun_dir) <= 0.0) return 0.0;
    uint lod = 0u;
    for (uint k = 1u; k < 3u; ++k)
        if (frame.shadow_lod_error[k] * 1.15 * frame.lod_scale / s.t < 0.5) lod = k;
    return trace_sun(s.hit, s.n, s.bias + frame.shadow_lod_error[lod] * 1.15 * 1.5, 1u << lod);
}
#endif

// Shadows are traced at half resolution by shadow.comp: one ray per two
// by two pixels, stored with the distance it was traced at.
layout(set = 0, binding = 20, scalar) buffer ShadowMask { vec2 shadow_mask[]; };

uvec2 shadow_size() {
    return uvec2((frame.width + 1u) / 2u, (frame.height + 1u) / 2u);
}
