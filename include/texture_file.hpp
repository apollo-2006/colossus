#pragma once
// a model's textures as streamed (.ctex), one per material: every mip level cut into tiles,
// bc1 compressed, so the viewer keeps only the tiles it samples (viewer/streamer.hpp streams
// them like geometry pages).
//
// a tile is 128 x 128 texels: 120 of the level and a 4 texel border, the level's texels
// around it (clamped at its edges, or wrapped round them for a repeating texture), so
// bilinear filtering never reads past a tile. levels halve, rounded up, until one tile holds
// a level. tile (x, y) of level l+1 holds the coarser copy of tiles (2x .. 2x+1, 2y .. 2y+1)
// of level l: its dependency.
//
// a texture is colour (bc1, srgb) or one channel of data (bc4, linear: a normal map's x or y,
// a roughness). bc4 blocks are bc1's size, so every tile is tile_bytes either way. a material
// names up to four textures: colour, normal x, normal y (opengl's convention: +y up the
// image), roughness.
//
// layout (CTEXv003): magic, texture count, the model's texture coordinate range (u min, v min,
// extent: the pages store fractions of it, paged_file.hpp), material count and per material
// its four texture indices (UINT32_MAX for none), then per texture width, height, level count,
// flags (repeat, double sided, one channel), and per level tiles_x, tiles_y, first tile; tiles
// from 16-byte aligned data_offset, tile_bytes each, texture by texture, level by level.
// CTEXv002 files (no material table: material k is colour texture k) still load.
#include "math.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

constexpr uint32_t tile_texels = 128, tile_border = 4, tile_payload = tile_texels - 2 * tile_border;
constexpr uint32_t tile_bytes = tile_texels * tile_texels / 2;  // bc1: 8 bytes per 4 x 4

struct image {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> rgb;  // srgb, rows top down
};

// binary ppm (p6, 8 bits).
image load_ppm(const std::string& path);

struct texture_level {
    uint32_t width, height, tiles_x, tiles_y, first_tile;
};

struct texture_entry {
    uint32_t width = 0, height = 0;
    bool repeat = false;
    bool double_sided = false;  // its material's: seen from both sides (no back face culling)
    bool single = false;        // one channel, linear (bc4); else colour (bc1)
    std::vector<texture_level> levels;  // first tiles counted across all the file's textures
};

// a material's textures: colour, normal x, normal y, roughness; UINT32_MAX for none.
enum texture_kind { kind_colour, kind_normal_x, kind_normal_y, kind_roughness, texture_kinds };
using material_textures = std::array<uint32_t, texture_kinds>;

struct texture_info {
    std::vector<texture_entry> textures;
    std::vector<material_textures> materials;
    vec2 uv_min;
    float uv_extent = 1;
    uint64_t data_offset = 0;
    uint32_t tile_count() const;
    // the tile one level coarser over the same texels, or UINT32_MAX for a last level's.
    uint32_t parent(uint32_t tile) const;
};

// a texture to write: colour from an image, or one channel of it (single).
struct texture_source {
    image source;
    bool single = false;
    uint8_t channel = 0;  // single: which of r, g, b
    uint8_t flags = 0;    // bit 0 repeat, bit 1 double sided
};

// mips (box filtered, colour in linear light), tiles, bc1 or bc4, for each texture, and the
// materials' table.
void save_textures(const std::vector<texture_source>& textures, const std::vector<material_textures>& materials,
                   vec2 uv_min, float uv_extent, const std::string& path);
// colour only, material k texture k. flags: bit 0 repeat, bit 1 double sided.
void save_textures(const std::vector<image>& images, const std::vector<uint8_t>& flags, vec2 uv_min, float uv_extent,
                   const std::string& path);
// one clamped texture over coordinates 0 to 1.
void save_texture(const image& base, const std::string& path);
texture_info load_texture_info(const std::string& path);

// a 4 x 4 block of rgb texels to bc1, and back (rgb per texel), for tests.
void encode_bc1(const uint8_t* rgb, uint8_t* out);
void decode_bc1(const uint8_t* block, uint8_t* rgb);
// a 4 x 4 block of single values to bc4, and back.
void encode_bc4(const uint8_t* v, uint8_t* out);
void decode_bc4(const uint8_t* block, uint8_t* v);
