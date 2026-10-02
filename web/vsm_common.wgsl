// virtual shadow maps, as vsm.glsl: the sun's depth in a clipmap of VSM_LEVELS
// levels, each a window of VSM_WINDOW x VSM_WINDOW pages of VSM_PAGE x VSM_PAGE
// texels, world-anchored, toroidal. physical pages only where pixels need them,
// kept across frames, rendered when new or invalid, in two layers: still and
// moving.
//
// shared by vsm.wgsl and compute.wgsl's shading. buffers are flat word arrays:
//
// entries: per slot 4 words (tag, physical page, frame last needed, flags);
// then per physical page 2 (owner slot, frame last used).
// lists: VSM_HEADER words cleared each frame (vsm.wgsl), then requests, free,
// evictable (VSM_SLOTS each), render list (twice that), moving instances.
// atlas: depths, two layers of VSM_SIDE x VSM_SIDE physical pages, page after page, tiled (vsm_atlas_index).

const VSM_LEVELS = 14u;
const VSM_WINDOW = 32u;
const VSM_PAGE = 128u;
override VSM_SIDE: u32 = 32u;  // physical pages a side: renderer.js sets it from the adapter's limits
const VSM_TEXEL0 = 1.0 / 4096.0;  // fine enough for close-ups on surfaces oblique to sun and camera
const VSM_SLOTS = VSM_LEVELS * VSM_WINDOW * VSM_WINDOW;
const VSM_NONE = 0xffffffffu;
const VSM_DIRTY = 1u;        // something moved over it: redraw the moving layer
const VSM_PROVISIONAL = 2u;  // drawn before its geometry loaded: redraw the still layer
const VSM_HAS_MOVING = 4u;   // its moving layer holds something; else lookups skip it
const VSM_STILL = 0u;
const VSM_MOVING = 1u;
const VSM_PHYS = 4u * VSM_SLOTS;  // physical page records start

const VSM_RECTS = 8u;        // in the lists header: per layer and level, 4 words
const VSM_MASKS = VSM_RECTS + 8u * VSM_LEVELS;  // per layer, level, slot row: a word
const VSM_HEADER = VSM_MASKS + 2u * VSM_LEVELS * VSM_WINDOW;
const VSM_REQUESTS = VSM_HEADER;
const VSM_UNOWNED = VSM_REQUESTS + VSM_SLOTS;
const VSM_EVICTABLE = VSM_UNOWNED + VSM_SLOTS;  // stale, then from half way recent
const VSM_RENDER = VSM_EVICTABLE + VSM_SLOTS;
const VSM_MOVING_LIST = VSM_RENDER + 2u * VSM_SLOTS;
const VSM_STALE = 8u;  // frames unneeded before a page is evicted ahead of recent ones (taa's cycle)

const FLAG_MOVING = 256u;  // some instances move

// light frame: x, y across the sun, z toward it (compute.wgsl's SUN_DIR).
const VSM_SUN = normalize(vec3f(0.75, 0.5, 0.3));
const VSM_X = normalize(cross(vec3f(0.0, 1.0, 0.0), VSM_SUN));
const VSM_Y = cross(VSM_SUN, VSM_X);

fn vsm_light_space(p: vec3f) -> vec3f {
  return vec3f(dot(p, VSM_X), dot(p, VSM_Y), dot(p, VSM_SUN));
}

fn vsm_texel(level: u32) -> f32 {
  return VSM_TEXEL0 * f32(1u << level);
}

fn vsm_center(level: u32) -> vec2i {
  return vec2i(floor(vsm_light_space(frame.origin.xyz).xy / (vsm_texel(level) * f32(VSM_PAGE))));
}

fn vsm_in_window(level: u32, page: vec2i) -> bool {
  let d = page - vsm_center(level);
  let half = i32(VSM_WINDOW / 2u);
  return all(d >= vec2i(-half)) && all(d < vec2i(half));
}

fn vsm_slot(level: u32, page: vec2i) -> u32 {
  let s = page & vec2i(i32(VSM_WINDOW) - 1);
  return level * VSM_WINDOW * VSM_WINDOW + u32(s.y) * VSM_WINDOW + u32(s.x);
}

fn vsm_tag(page: vec2i) -> u32 {
  return (u32(page.x) & 0xffffu) | (u32(page.y) << 16u);
}

fn vsm_slot_page(slot: u32) -> vec2i {
  let level = slot / (VSM_WINDOW * VSM_WINDOW);
  let s = vec2i(i32(slot % VSM_WINDOW), i32((slot / VSM_WINDOW) % VSM_WINDOW));
  let lo = vsm_center(level) - vec2i(i32(VSM_WINDOW / 2u));
  return lo + ((s - lo) & vec2i(i32(VSM_WINDOW) - 1));
}

fn vsm_level_for(t: f32) -> u32 {
  let footprint = t / frame.lod_scale;
  return u32(clamp(ceil(log2(max(footprint / VSM_TEXEL0, 1.0))), 0.0, f32(VSM_LEVELS - 1u)));
}

// the same for a surface lit at cosine nl: a texel stretches over it by 1 / nl, so grazing surfaces
// take finer levels (up to two).
fn vsm_level_for_lit(t: f32, nl: f32) -> u32 {
  return vsm_level_for(t * clamp(nl, 0.25, 1.0));
}

fn vsm_sortable(z: f32) -> u32 {
  let b = bitcast<u32>(z);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn vsm_unsortable(u: u32) -> f32 {
  return bitcast<f32>(select(~u, u & 0x7fffffffu, (u & 0x80000000u) != 0u));
}

// page after page, each in 8x8 tiles of 256 bytes: a small triangle's texels share a tile or two.
fn vsm_atlas_index(phys: u32, in_page: vec2u, layer: u32) -> u32 {
  let page = layer * VSM_SIDE * VSM_SIDE + phys;
  let tile = in_page / 8u;
  let in_tile = in_page % 8u;
  return page * (VSM_PAGE * VSM_PAGE) + (tile.y * (VSM_PAGE / 8u) + tile.x) * 64u + in_tile.y * 8u + in_tile.x;
}

fn vsm_mask_word(layer: u32, slot: u32) -> u32 {
  return VSM_MASKS + layer * VSM_LEVELS * VSM_WINDOW + slot / VSM_WINDOW;
}
