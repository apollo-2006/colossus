#pragma once
// Writes 8-bit RGB PNG files with no image library: a zlib stream of
// stored (uncompressed) deflate blocks, which every reader accepts.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace png {

inline uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc = 0) {
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

inline bool write_rgb(const std::string& path, int w, int h, const std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> raw;  // Each row: filter byte 0, then the pixels
    raw.reserve(size_t(h) * (1 + 3 * size_t(w)));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + size_t(y) * 3 * w, rgb.begin() + size_t(y + 1) * 3 * w);
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    for (size_t at = 0; at < raw.size() || at == 0; at += 65535) {
        const size_t n = std::min<size_t>(65535, raw.size() - at);
        z.push_back(at + n >= raw.size() ? 1 : 0);
        z.push_back(n & 255); z.push_back(n >> 8);
        z.push_back(~n & 255); z.push_back((~n >> 8) & 255);
        z.insert(z.end(), raw.begin() + at, raw.begin() + at + n);
        if (n == 0) break;
    }
    uint32_t a = 1, b = 0;  // Adler-32
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (int s = 24; s >= 0; s -= 8) z.push_back((((b << 16) | a) >> s) & 255);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
        uint8_t len[4] = {uint8_t(data.size() >> 24), uint8_t(data.size() >> 16), uint8_t(data.size() >> 8), uint8_t(data.size())};
        std::fwrite(len, 1, 4, f);
        std::vector<uint8_t> td(type, type + 4);
        td.insert(td.end(), data.begin(), data.end());
        std::fwrite(td.data(), 1, td.size(), f);
        const uint32_t crc = crc32(td.data(), td.size());
        uint8_t c[4] = {uint8_t(crc >> 24), uint8_t(crc >> 16), uint8_t(crc >> 8), uint8_t(crc)};
        std::fwrite(c, 1, 4, f);
    };
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::fwrite(sig, 1, 8, f);
    std::vector<uint8_t> ihdr = {uint8_t(w >> 24), uint8_t(w >> 16), uint8_t(w >> 8), uint8_t(w),
                                 uint8_t(h >> 24), uint8_t(h >> 16), uint8_t(h >> 8), uint8_t(h), 8, 2, 0, 0, 0};
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    return std::fclose(f) == 0;
}

}  // namespace png
