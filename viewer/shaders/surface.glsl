// what is under a pixel, shared by shade.comp and shadow.comp, which must
// agree: the visible triangle refetched, the pixel ray meeting its plane for
// barycentrics, distance from depth. nothing drawn is ground or sky.


const uint surface_sky = 0u, surface_ground = 1u, surface_object = 2u;

struct Surface {
    uint kind;
    vec3 dir;     // from the camera through the pixel
    float t;      // distance along dir
    vec3 hit;
    vec3 n;       // facing the camera
    bool behind;  // back of the surface: inside a fold or through a hole
    float bias;   // shadow ray start offset, for the drawn surface's error
    uint id;      // in the visibility buffer
    uvec2 vc;     // (instance, cluster)
    vec3 p0, p1, p2;
    uint base;       // the cluster's page
    uvec3 corners;   // the triangle's vertices in the cluster
    vec2 bary;       // weights of corners 1 and 2
};

// weights of a triangle's corners 1 and 2 where the ray from the camera along dir meets its
// plane (moller-trumbore), unclamped.
vec2 plane_bary(vec3 p0, vec3 p1, vec3 p2, vec3 dir) {
    const vec3 e1 = p1 - p0, e2 = p2 - p0;
    const vec3 pv = cross(dir, e2);
    const float det = dot(e1, pv);
    const vec3 tv = frame.origin.xyz - p0;
    const float inv = abs(det) > 1e-20 ? 1.0 / det : 0.0;
    return vec2(dot(tv, pv), dot(dir, cross(tv, e1))) * inv;
}

vec3 pixel_ray_at(vec2 p) {
    const vec2 ndc = p / vec2(frame.width, frame.height) * 2.0 - 1.0;
    const vec4 near_point = frame.inv_view_proj * vec4(ndc, 1.0, 1.0);
    return normalize(near_point.xyz / near_point.w - frame.origin.xyz);
}

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
    const Cluster c = load_cluster(s.vc.y);
    const uint base = page_table[c.group];
    const uint packed = cluster_triangle(base, c, s.id & 127u);
    const uint i0 = packed & 255u, i1 = (packed >> 8) & 255u, i2 = (packed >> 16) & 255u;
    const vec4 bounds = meshes[inst.mesh].bounds;
    s.p0 = to_world(inst, drawn_vertex(inst, base, c, i0));
    s.p1 = to_world(inst, drawn_vertex(inst, base, c, i1));
    s.p2 = to_world(inst, drawn_vertex(inst, base, c, i2));

    // ray against the triangle's plane (moller-trumbore, no bounds checks: the
    // rasterizer said it hits).
    const vec3 e1 = s.p1 - s.p0, e2 = s.p2 - s.p0;
    const vec3 pv = cross(s.dir, e2);
    const float det = dot(e1, pv);
    const vec3 tv = origin - s.p0;
    const float inv = abs(det) > 1e-20 ? 1.0 / det : 0.0;
    const vec3 qv = cross(tv, e1);
    // clamped: at a silhouette the triangle can be edge-on and the plane hit
    // far outside.
    const float bu = clamp(dot(tv, pv) * inv, 0.0, 1.0);
    const float bv = clamp(dot(s.dir, qv) * inv, 0.0, 1.0 - bu);
    s.base = base;
    s.corners = uvec3(i0, i1, i2);
    s.bary = vec2(bu, bv);
    // distance from depth, which is exact.
    const vec4 center = frame.inv_view_proj * vec4(0.0, 0.0, 1.0, 1.0);
    const vec3 forward = normalize(center.xyz / center.w - origin);
    s.t = frame.near_z / uintBitsToFloat(uint(v >> 32)) / dot(s.dir, forward);
    s.hit = origin + s.dir * s.t;
    vec3 n0 = cluster_normal(base, c, i0), n1 = cluster_normal(base, c, i1), n2 = cluster_normal(base, c, i2);
    if (skinned(inst)) {  // each vertex's normal turned by its joints
        n0 = skin_dir(inst, base, c, i0, n0);
        n1 = skin_dir(inst, base, c, i1, n1);
        n2 = skin_dir(inst, base, c, i2, n2);
    }
    const vec3 model_n = n0 * (1.0 - bu - bv) + n1 * bu + n2 * bv;
    s.n = normalize(to_world_dir(inst, deform_normal(inst, bounds, from_world(inst, s.hit), model_n, frame.time)));
    s.behind = dot(s.n, s.dir) > 0.0;
    if (s.behind) s.n = -s.n;
    // the drawn cut strays up to this cluster's error; trace_surface() adds the
    // shadow copy's.
    s.bias = c.lod_error * inst.scale * 1.5 + 1e-4 + s.t * 2e-5;
    return s;
}

#ifdef RAY_QUERY
#extension GL_EXT_ray_query : require
layout(set = 0, binding = 18) uniform accelerationStructureEXT shadow_scene;   // still instances
layout(set = 0, binding = 24) uniform accelerationStructureEXT shadow_moving;  // moving ones, refitted every frame

bool hits(accelerationStructureEXT scene, vec3 start, uint mask) {
    rayQueryEXT q;
    rayQueryInitializeEXT(q, scene,
                          gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT, mask,
                          start, 0.0, sun_dir, (frame.scene_top - start.y) / sun_dir.y);
    while (rayQueryProceedEXT(q)) {
    }
    return rayQueryGetIntersectionTypeEXT(q, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

// 1 if the sun reaches p, else 0. starts `bias` above the surface (the shadow
// copy may stand that far above). back faces skipped, so where the drawn
// surface dips inside the copy the ray leaves through its back (models wind
// counterclockwise from outside).
//
// nothing stands above frame.scene_top, so rays stop there: no lost shadow, far
// less traversal.
float trace_sun(vec3 p, vec3 n, float bias, uint mask) {
    const vec3 start = p + n * bias;
    if (start.y >= frame.scene_top) return 1.0;
    if (hits(shadow_scene, start, mask)) return 0.0;
    return (frame.flags & flag_moving) != 0u && hits(shadow_moving, start, mask) ? 0.0 : 1.0;
}

// lit test: traced against the coarsest shadow copy whose error is under half a
// pixel at this distance, clearing that error.
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

// half resolution shadows from shadow.comp: a ray per 2x2 pixels, with its
// distance.
layout(set = 0, binding = 20, scalar) buffer ShadowMask { vec2 shadow_mask[]; };

uvec2 shadow_size() {
    return uvec2((frame.width + 1u) / 2u, (frame.height + 1u) / 2u);
}
