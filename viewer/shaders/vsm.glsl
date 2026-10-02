// virtual shadow maps: the sun's depth in a clipmap of vsm_levels levels around
// the camera. level l's texels are vsm_texel0 * 2^l across; each level is a
// window of vsm_window x vsm_window pages of vsm_page x vsm_page texels,
// anchored to the world, held toroidally: page (x, y) lives in slot (x mod
// vsm_window, y mod vsm_window).
//
// physical pages back only pages a pixel needs (vsm_mark.comp) and persist
// until the window leaves them or they are evicted. two layers: still
// instances, rendered when new (or drawn before geometry loaded); moving
// instances, rendered again wherever one moved. vsm_alloc.comp lists the work,
// the vsm culling passes draw the hierarchy at each level's texel detail,
// shading reads the nearer layer.

const uint vsm_levels = 12u;
const uint vsm_window = 32u;   // pages a side, per level
const uint vsm_page = 128u;    // texels a side, per page
const float vsm_texel0 = 1.0 / 1024.0;
const uint vsm_slots = vsm_levels * vsm_window * vsm_window;
const uint vsm_none = 0xffffffffu;

// one per slot: which page, and where.
struct VsmEntry {
    uint tag;     // page coordinates, x | y << 16 (16 bits each, two's complement)
    uint phys;    // physical page, or vsm_none
    uint needed;  // last frame a pixel needed it
    uint flags;   // vsm_dirty, vsm_provisional
};
const uint vsm_dirty = 1u;        // something moved over it: redraw the moving layer
const uint vsm_provisional = 2u;  // drawn before its geometry loaded: redraw the still layer
const uint vsm_has_moving = 4u;   // its moving layer holds something; else lookups skip it
const uint vsm_still = 0u, vsm_moving = 1u;  // layers

// one per physical page.
struct VsmPhys {
    uint owner;      // slot it backs, or vsm_none
    uint last_used;  // last frame its page was needed
};

layout(set = 0, binding = 27, scalar) buffer VsmEntries { VsmEntry vsm_entries[]; };
layout(set = 0, binding = 28, scalar) buffer VsmPhysPages { VsmPhys vsm_phys[]; };
// per-frame lists; the first vsm_lists_header words are cleared each frame.
layout(set = 0, binding = 29, scalar) buffer VsmLists {
    uint vsm_request_count;    // needed, not resident
    uint vsm_unowned_count;    // free physical pages
    uint vsm_evictable_count;  // physical pages not needed this frame
    uint vsm_render_count;        // slot layers to render
    uint vsm_rendered_levels[2];  // per layer, bit l: level l renders
    uint vsm_overflow;            // needed pages left without a physical page
    uint vsm_pad0;
    ivec4 vsm_render_rect[2 * vsm_levels];  // per layer and level: rendered pages span [xy, zw]
    uint vsm_render_mask[2 * vsm_levels * vsm_window];  // per layer, level, slot row: bit x if slot (x, row) renders
    // requests, unowned, evictable: vsm_slots each. then the render list, slot
    // | layer << 31, twice that.
    uint vsm_list[];
};
const uint vsm_lists_header = 8u + 8u * vsm_levels + 2u * vsm_levels * vsm_window;

// atlas: physical page p at (p mod side, p / side) * vsm_page, side =
// frame.vsm_atlas_side. depths as sortable uints (vsm_sortable): larger is
// nearer the sun, 0 empty.
layout(set = 0, binding = 30, scalar) buffer VsmAtlas { uint vsm_atlas[]; };

// light frame: x, y across the sun, z toward it.
const vec3 vsm_z = sun_dir;
const vec3 vsm_x = normalize(cross(vec3(0.0, 1.0, 0.0), sun_dir));
const vec3 vsm_y = cross(vsm_z, vsm_x);

vec3 vsm_light_space(vec3 p) {
    return vec3(dot(p, vsm_x), dot(p, vsm_y), dot(p, vsm_z));
}

float vsm_texel(uint level) {
    return vsm_texel0 * float(1u << level);
}

// page a level's window centres on, from the camera.
ivec2 vsm_center(uint level) {
    return ivec2(floor(vsm_light_space(frame.origin.xyz).xy / (vsm_texel(level) * float(vsm_page))));
}

bool vsm_in_window(uint level, ivec2 page) {
    const ivec2 d = page - vsm_center(level);
    return all(greaterThanEqual(d, ivec2(-int(vsm_window) / 2))) && all(lessThan(d, ivec2(int(vsm_window) / 2)));
}

uint vsm_slot(uint level, ivec2 page) {
    const ivec2 s = page & ivec2(vsm_window - 1u);  // two's complement: a positive modulo
    return level * vsm_window * vsm_window + uint(s.y) * vsm_window + uint(s.x);
}

uint vsm_tag(ivec2 page) {
    return (uint(page.x) & 0xffffu) | (uint(page.y) << 16);
}

// page a slot holds in the current window.
ivec2 vsm_slot_page(uint slot) {
    const uint level = slot / (vsm_window * vsm_window);
    const ivec2 s = ivec2(slot % vsm_window, (slot / vsm_window) % vsm_window);
    const ivec2 lo = vsm_center(level) - ivec2(vsm_window / 2u);
    // page in [lo, lo + window) congruent to s.
    return lo + ((s - lo) & ivec2(vsm_window - 1u));
}

// level whose texels match a pixel's footprint at distance t.
uint vsm_level_for(float t) {
    const float footprint = t / frame.lod_scale;
    return uint(clamp(ceil(log2(max(footprint / vsm_texel0, 1.0))), 0.0, float(vsm_levels - 1u)));
}

uint vsm_sortable(float z) {
    const uint b = floatBitsToUint(z);
    return (b & 0x80000000u) != 0u ? ~b : b | 0x80000000u;
}

float vsm_unsortable(uint u) {
    return uintBitsToFloat((u & 0x80000000u) != 0u ? u & 0x7fffffffu : ~u);
}

uint vsm_atlas_index(uint phys, uvec2 texel_in_page, uint layer) {
    const uint side = frame.vsm_atlas_side;
    const uvec2 at = uvec2(phys % side, phys / side) * vsm_page + texel_in_page;
    return (layer * side * vsm_page + at.y) * side * vsm_page + at.x;
}

uint vsm_mask_word(uint layer, uint slot) {
    return layer * vsm_levels * vsm_window + slot / vsm_window;
}

bool vsm_renders(uint layer, uint slot) {
    return (vsm_render_mask[vsm_mask_word(layer, slot)] & (1u << (slot % vsm_window))) != 0u;
}

const uint vsm_big_capacity = 65536u;

// moving instances, for invalidation.
layout(set = 0, binding = 34, scalar) readonly buffer MovingInstances { uint moving_instances[]; };

// culling lists, separate from the camera's.
layout(set = 0, binding = 31, scalar) buffer VsmWork {
    uint vsm_work_count;
    uint vsm_big_count;      // instances with many work items, for vsm_expand.comp
    uint vsm_visible_count;
    uint vsm_pad2;
    uvec2 vsm_work[];        // (instance | level << 24 | layer << 28, first cluster); big list from max_work_items, two entries each
};
layout(set = 0, binding = 32, scalar) buffer VsmVisible { uvec2 vsm_visible[]; };  // (instance | level << 24 | layer << 28, cluster)
layout(set = 0, binding = 33, scalar) buffer VsmArgs {
    uvec4 vsm_instance_args;
    uvec4 vsm_clear_args;
    uvec4 vsm_expand_args;
    uvec4 vsm_cull_args;
    uvec4 vsm_raster_args;
};

// pages to render on a level, decoded.
ivec4 vsm_rendered(uint layer, uint level) {
    const ivec4 r = vsm_render_rect[layer * vsm_levels + level];
    return ivec4((1 << 20) - r.x, (1 << 20) - r.y, r.z - (1 << 20), r.w - (1 << 20));
}

// whether a light-space sphere reaches a page being rendered. big spheres check
// bounds only.
bool vsm_touches(uint layer, uint level, vec2 center, float radius) {
    const float size = vsm_texel(level) * float(vsm_page);
    const ivec4 r = vsm_rendered(layer, level);
    const ivec2 lo = max(ivec2(floor((center - radius) / size)), r.xy);
    const ivec2 hi = min(ivec2(floor((center + radius) / size)), r.zw);
    if (any(greaterThan(lo, hi))) return false;
    if (hi.x - lo.x > 7 || hi.y - lo.y > 7) return true;
    for (int y = lo.y; y <= hi.y; ++y)
        for (int x = lo.x; x <= hi.x; ++x) {
            const ivec2 page = ivec2(x, y);
            if (!vsm_in_window(level, page)) continue;
            if (vsm_renders(layer, vsm_slot(level, page))) return true;
        }
    return false;
}

// soft shadows: contact hardening (pcss). the sun is drawn 1.5 degrees wide, wider than the real
// 0.53, so penumbras show at statue scale.
const float vsm_penumbra_per_unit = 0.0262;  // 2 tan(0.75 degrees): penumbra width per unit of gap
const float vsm_max_penumbra = 12.0;         // texels, radius; marking covers it
const uint vsm_search_taps = 6u, vsm_filter_taps = 8u;

// stored depth at a texel (nearer layer), or false if its page has no physical page. taps mostly
// share the centre's page: its physical page (cached_phys) skips the entry read.
bool vsm_texel_depth(uint level, ivec2 at, ivec2 cached_page, uint cached_phys, bool cached_moving, out uint stored) {
    const ivec2 page = at >> 7;
    uint phys = cached_phys;
    bool moving = cached_moving;
    if (page != cached_page) {
        if (!vsm_in_window(level, page)) return false;
        const VsmEntry e = vsm_entries[vsm_slot(level, page)];
        if (e.tag != vsm_tag(page)) return false;
        phys = e.phys;
        moving = (e.flags & vsm_has_moving) != 0u;
    }
    if (phys == vsm_none) return false;
    const uvec2 in_page = uvec2(at & ivec2(int(vsm_page) - 1));
    stored = vsm_atlas[vsm_atlas_index(phys, in_page, vsm_still)];
    if (moving) stored = max(stored, vsm_atlas[vsm_atlas_index(phys, in_page, vsm_moving)]);
    return true;
}

// vogel disk point k of n, rotated by angle.
vec2 vogel(uint k, uint n, float angle) {
    const float r = sqrt((float(k) + 0.5) / float(n));
    const float a = float(k) * 2.39996323 + angle;
    return r * vec2(cos(a), sin(a));
}

// sunlight at p (normal n, distance t), 0 to 1, from the pages at the pixel's level. offset along
// the normal and biased a couple of texels, covering both surfaces' lod error. a blocker search
// sizes the penumbra from the gap to the average blocker; under a texel, 2x2 bilinear taps,
// else vogel taps rotated by `noise` (taa averages them). a tap without a physical page sends
// the lookup up a level.
float vsm_lookup(vec3 p, vec3 n, float t, float noise, out uint used) {
    for (uint level = vsm_level_for(t); level < vsm_levels; ++level) {
        used = level;
        const float texel = vsm_texel(level);
        const vec3 lp = vsm_light_space(p + n * (2.0 * texel));
        const float receiver = lp.z + 1.5 * texel;
        const vec2 centre = lp.xy / texel;
        const float angle = noise * 6.2831853;
        const ivec2 home = ivec2(floor(centre)) >> 7;
        uint home_phys = vsm_none;
        bool home_moving = false;
        if (vsm_in_window(level, home)) {
            const VsmEntry e = vsm_entries[vsm_slot(level, home)];
            if (e.tag == vsm_tag(home)) {
                home_phys = e.phys;
                home_moving = (e.flags & vsm_has_moving) != 0u;
            }
        }
        bool complete = true;
        float blockers = 0.0, count = 0.0;
        const bool soft = (frame.flags & flag_soft_shadows) != 0u;
        for (uint k = 0u; k < vsm_search_taps && complete && soft; ++k) {
            uint stored;
            complete = vsm_texel_depth(level, ivec2(floor(centre + vogel(k, vsm_search_taps, angle) * vsm_max_penumbra)), home, home_phys, home_moving, stored);
            if (complete && stored != 0u && vsm_unsortable(stored) > receiver) {
                blockers += vsm_unsortable(stored);
                count += 1.0;
            }
        }
        if (!complete) continue;
        if (soft && count == 0.0) return 1.0;  // nothing nearby stands between it and the sun
        const float penumbra = soft ? (blockers / count - lp.z) * vsm_penumbra_per_unit * 0.5 / texel : 0.0;
        float lit = 0.0;
        if (penumbra < 1.0) {
            const vec2 f = centre - 0.5;
            const ivec2 base = ivec2(floor(f));
            const vec2 w = f - vec2(base);
            for (int k = 0; k < 4 && complete; ++k) {
                uint stored;
                complete = vsm_texel_depth(level, base + ivec2(k & 1, k >> 1), home, home_phys, home_moving, stored);
                const bool open = stored == 0u || vsm_unsortable(stored) <= receiver;
                const float weight = ((k & 1) != 0 ? w.x : 1.0 - w.x) * ((k >> 1) != 0 ? w.y : 1.0 - w.y);
                lit += open ? weight : 0.0;
            }
        } else {
            const float radius = min(penumbra, vsm_max_penumbra);
            for (uint k = 0u; k < vsm_filter_taps && complete; ++k) {
                uint stored;
                complete = vsm_texel_depth(level, ivec2(floor(centre + vogel(k, vsm_filter_taps, angle + 1.0) * radius)), home, home_phys, home_moving, stored);
                lit += stored == 0u || vsm_unsortable(stored) <= receiver ? 1.0 : 0.0;
            }
            lit /= float(vsm_filter_taps);
        }
        if (complete) return lit;
    }
    used = vsm_levels;
    return 1.0;
}

float vsm_shadow(vec3 p, vec3 n, float t, float noise) {
    uint used;
    return vsm_lookup(p, n, t, noise, used);
}
