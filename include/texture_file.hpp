#pragma once
// a model's texture as streamed (.ctex): every mip level cut into tiles, bc1 compressed, so
// the viewer keeps only the tiles it samples, in an atlas (viewer/streamer.hpp streams them
// like geometry pages).
//
// a tile is 128 x 128 texels: 120 of the level and a 4 texel border, the level's texels
// around it (clamped at its edges), so bilinear filtering never reads past a tile. levels
// halve, rounded up, until one tile holds a level. tile (x, y) of level l+1 holds the
// coarser copy of tiles (2x .. 2x+1, 2y .. 2y+1) of level l: its dependency.
//
// layout: magic, width, height, level count, then per level tiles_x, tiles_y, first tile;
// tiles from 16-byte aligned data_offset, tile_bytes each, in level order.
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

struct texture_info {
    uint32_t width = 0, height = 0;
    std::vector<texture_level> levels;
    uint64_t data_offset = 0;
    uint32_t tile_count() const { return levels.empty() ? 0 : levels.back().first_tile + levels.back().tiles_x * levels.back().tiles_y; }
    // the tile one level coarser over the same texels, or UINT32_MAX for the last level's.
    uint32_t parent(uint32_t tile) const;
};

// mips (box filtered in linear light), tiles, bc1.
void save_texture(const image& base, const std::string& path);
texture_info load_texture_info(const std::string& path);

// a 4 x 4 block of rgb texels to bc1, and back (rgb per texel), for tests.
void encode_bc1(const uint8_t* rgb, uint8_t* out);
void decode_bc1(const uint8_t* block, uint8_t* rgb);
