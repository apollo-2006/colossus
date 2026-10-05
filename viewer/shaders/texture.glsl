// a model's texture, streamed as tiles into the page pool (include/texture_file.hpp): bc1
// decoded here, bilinear within a tile (its border holds the neighbours), trilinear across
// levels. a missing tile is asked for and its nearest resident coarser copy drawn: tiles are
// resident only while their coarser copies are (viewer/streamer.hpp), and the coarsest is
// pinned.

const uint tile_texels = 128u, tile_border = 4u, tile_payload = 120u;

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

vec3 unpack565(uint v) {
    return vec3(float(v >> 11u), float((v >> 5u) & 63u), float(v & 31u)) / vec3(31.0, 63.0, 31.0);
}

// texel t (0 to 127 each way) of the tile at word `at`, linear.
vec3 tile_texel(uint at, uvec2 t) {
    const uint block = at + 2u * ((t.y >> 2u) * (tile_texels / 4u) + (t.x >> 2u));
    const uint ends = pool[block], bits = pool[block + 1u];
    const uint c0 = ends & 0xffffu, c1 = ends >> 16u;
    const uint index = (bits >> (2u * ((t.y & 3u) * 4u + (t.x & 3u)))) & 3u;
    const vec3 a = unpack565(c0), b = unpack565(c1);
    vec3 c = index == 0u ? a : index == 1u ? b : c0 > c1 ? (index == 2u ? (2.0 * a + b) / 3.0 : (a + 2.0 * b) / 3.0)
                                                         : (index == 2u ? (a + b) * 0.5 : vec3(0.0));
    return srgb_to_linear(c);
}

struct TextureLevel {
    uvec2 size, tiles;
    uint first;  // its first tile's page
};

TextureLevel texture_level(uvec4 tex, uint level) {
    TextureLevel l;
    l.size = tex.zw;
    l.first = tex.x;
    for (uint k = 0u;; ++k) {
        l.tiles = (l.size + tile_payload - 1u) / tile_payload;
        if (k == level) return l;
        l.first += l.tiles.x * l.tiles.y;
        l.size = (l.size + 1u) / 2u;
    }
}

// bilinear at uv from the finest resident level from `level` up. a missing `level` tile is
// asked for, more urgently the coarser the copy drawn instead (in the geometry requests'
// units: as if each level coarser were a pixel of error more, doubling).
vec3 sample_resident(uvec4 tex, vec2 uv, uint level) {
    uint wanted = NO_PAGE;
    const uint levels = tex.y & 255u;
    // a repeating texture wraps (its tile borders wrap round too); else clamps.
    const vec2 st = (tex.y & 256u) != 0u ? fract(uv) : clamp(uv, 0.0, 1.0);
    for (uint k = level; k < levels; ++k) {
        const TextureLevel l = texture_level(tex, k);
        // the texel left of and above uv's centre, and the weights to the next. at is at least
        // -0.5, so b at least -1: that texel is the tile border's.
        const vec2 at = st * vec2(l.size) - 0.5;
        const vec2 base = floor(at), f = at - base;
        const ivec2 b = ivec2(base);
        const uvec2 tile = min(uvec2(max(b, ivec2(0))) / tile_payload, l.tiles - 1u);
        const uint page = l.first + tile.y * l.tiles.x + tile.x;
        const uint word = page_table[page];
        if (word == NO_PAGE) {
            if (k == level) wanted = page;
            continue;
        }
        if (wanted != NO_PAGE) request_page(wanted, frame.lod_threshold * exp2(float(k - level)));
        if (page_used[page] != frame.frame_index) page_used[page] = frame.frame_index;
        // inside the tile, border included: the four texels never leave it.
        const uvec2 t = uvec2(clamp(b - ivec2(tile * tile_payload) + int(tile_border), ivec2(0), ivec2(tile_texels - 2u)));
        return mix(mix(tile_texel(word, t), tile_texel(word, t + uvec2(1, 0)), f.x),
                   mix(tile_texel(word, t + uvec2(0, 1)), tile_texel(word, t + uvec2(1, 1)), f.x), f.y);
    }
    return vec3(1.0, 0.0, 1.0);  // unreachable: the coarsest tile is pinned
}

// trilinear: the level whose texels match the pixel footprint (uv derivatives across a
// pixel), and the next.
vec3 sample_texture(uvec4 tex, vec2 uv, vec2 duv_dx, vec2 duv_dy) {
    const vec2 size = vec2(tex.zw);
    const float footprint = max(length(duv_dx * size), length(duv_dy * size));
    const uint levels = tex.y & 255u;
    const float lod = clamp(log2(max(footprint, 1e-8)), 0.0, float(levels - 1u));
    const uint level = uint(lod);
    const float f = lod - float(level);
    const vec3 a = sample_resident(tex, uv, level);
    if (f < 1.0 / 64.0 || level + 1u >= levels) return a;
    return mix(a, sample_resident(tex, uv, level + 1u), f);
}
