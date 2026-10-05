// light from everything but the sun, shared by shading (shade.comp) and the indirect pass
// (ao.comp). include after vsm.glsl (and, with RAY_QUERY, surface.glsl).

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// aces() undone (its quadratic solved for x), for reading back what was drawn as light.
vec3 aces_inverse(vec3 y) {
    y = clamp(y, 0.0, 0.99);
    const vec3 a = 2.43 * y - 2.51, b = 0.59 * y - 0.03, c = 0.14 * y;
    return (-b - sqrt(max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a);
}

// per instance (Instance::material): albedo, roughness, metalness. surface detail (shade.comp)
// varies them; ray traced bounce light takes them plain.
struct Material {
    vec3 albedo;
    float roughness;
    float metallic;
};
const Material materials[6] = Material[](
    Material(vec3(0.56, 0.52, 0.47), 0.6, 0.0),   // plaster: the plain look
    Material(vec3(0.66, 0.64, 0.6), 0.3, 0.0),    // polished marble
    Material(vec3(0.62, 0.5, 0.38), 0.85, 0.0),   // sandstone
    Material(vec3(0.58, 0.38, 0.22), 0.35, 1.0),  // bronze
    Material(vec3(1.0, 0.7, 0.27), 0.28, 1.0),    // gold
    Material(vec3(0.2, 0.2, 0.22), 0.45, 0.0));   // dark granite

// the ground's colour and the light on it, as drawn: its radiance where it is in sun (lit 1) or
// shadow (lit 0). the grid lines aside.
#define ground_albedo (frame.ground.rgb)  // the scene's (0.41, 0.39, 0.36 by default)

vec3 sky(vec3 dir) {
    const float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3(0.66, 0.68, 0.71), vec3(0.3, 0.42, 0.62), pow(t, 0.7)) * 0.9;
}

vec3 ground_radiance(float lit, float ao) {
    return ground_albedo * (sun_color * max(sun_dir.y, 0.0) * 0.6 * lit + sky(vec3(0, 1, 0)) * 0.45 * ao);
}


// light arriving at p (normal n) from all but the sun, times occlusion: the sky above, and below
// the ground, lit as it is drawn. rays down a cosine lobe find where they land and whether that
// spot is in shadow, so the sunlit ground warms what faces it and the ground in a statue's own
// shadow does not. four rays a pixel, turned each frame for taa to average.
// the plain ambient: sky above, a fixed grey below.
vec3 plain_ambient(vec3 n, float ao) {
    return mix(vec3(0.24, 0.22, 0.2) * 0.42, sky(vec3(0.0, 1.0, 0.0)) * 0.42, n.y * 0.5 + 0.5) * ao;
}

// sunlight on a ground point, hard edged: from the shadow maps, or a ray at the coarsest shadow
// copy (a bounce needs no finer).
float ground_lit(vec3 g) {
    if ((frame.flags & flag_vsm) != 0u) return vsm_hard_shadow(g, vec3(0.0, 1.0, 0.0), length(g - frame.origin.xyz));
#ifdef RAY_QUERY
    return trace_sun(g, vec3(0.0, 1.0, 0.0), 1e-3 + frame.shadow_lod_error[2] * 1.15 * 1.5, 4u);
#else
    return 1.0;
#endif
}

vec3 ambient_light(vec3 p, vec3 n, float ao, float noise) {
    const vec3 up_light = sky(vec3(0.0, 1.0, 0.0)) * 0.42;
#ifdef RAY_QUERY
    const bool ground_shadows = true;
#else
    const bool ground_shadows = (frame.flags & flag_vsm) != 0u;
#endif
    if ((frame.flags & (flag_bounce | flag_shadows)) != (flag_bounce | flag_shadows) || !ground_shadows) return plain_ambient(n, ao);
    const vec3 t = normalize(abs(n.y) < 0.9 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0))), b = cross(n, t);
    vec3 sum = vec3(0.0);
    for (uint k = 0u; k < 4u; ++k) {
        // cosine weighted: a disc point lifted onto the hemisphere.
        const float r = sqrt((float(k) + 0.5) / 4.0), a = (float(k) * 2.39996 + noise * 6.2831853);
        const vec3 dir = normalize(t * (r * cos(a)) + b * (r * sin(a)) + n * sqrt(max(1.0 - r * r, 0.0)));
        if (dir.y >= 0.0 || p.y <= 0.0) {
            sum += up_light;
            continue;
        }
        const vec3 g = p + dir * (-p.y / dir.y);
        sum += ground_radiance(ground_lit(g), 1.0);
    }
    return sum * 0.25 * ao;
}

