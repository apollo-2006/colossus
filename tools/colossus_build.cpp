// colossus_build: reads a ply or obj model, builds its cluster lod hierarchy,
// writes a .cgeo for the viewer.
//
//     colossus_build models/lucy.ply models/lucy.cgeo --up-z
//     colossus_build in.obj out.cgeo --check     also check every cut for cracks
//     colossus_build in.ply out.cgeo --max-triangles 500000
//                                            keep only the levels from the finest
//                                            cut within 500k triangles up
//     colossus_build in.obj out.cgeo --texture t.ppm
//                                            the texture, if not the obj's map_Kd
//     colossus_build fox.gltf fox.cgeo --subdivide 5
//                                            loop subdivision, five times, first: a
//                                            low poly model made smooth and dense
//     colossus_build pine.gltf pine.cgeo --node pine_a --keep-scale
//                                            one node of a gltf, in its own units (metres)
//                                            rather than one unit across: for scenes
//     colossus_build in.ply out.cgeo --normal-weight 1
//                                            collapses also priced by the change of
//                                            normal: better shading per triangle (flip
//                                            5 to 10% lower), more triangles a pixel
//
// a textured model also writes out.ctex beside out.cgeo (include/texture_file.hpp), a texture
// per material. textures are read as binary ppm: one in another format is looked for under the
// same name ending .ppm (ffmpeg -i t.jpg t.ppm makes one). a material without one gets its
// colour. a skinned gltf also writes out.cskn: its
// skeleton, animations and per page skin bounds (include/skeleton.hpp).
#include "geometry_file.hpp"
#include "lod_check.hpp"
#include "paged_file.hpp"
#include "skeleton.hpp"
#include "texture_file.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    std::string in, out, texture, node;
    bool up_z = false, check = false;
    size_t max_triangles = 0;
    int subdivisions = 0;
    bool keep_scale = false;
    float normal_weight = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--up-z")) up_z = true;
        else if (!std::strcmp(argv[i], "--max-triangles") && i + 1 < argc) max_triangles = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--texture") && i + 1 < argc) texture = argv[++i];
        else if (!std::strcmp(argv[i], "--subdivide") && i + 1 < argc) subdivisions = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--node") && i + 1 < argc) node = argv[++i];
        else if (!std::strcmp(argv[i], "--keep-scale")) keep_scale = true;
        else if (!std::strcmp(argv[i], "--normal-weight") && i + 1 < argc) normal_weight = std::stof(argv[++i]);
        else if (in.empty()) in = argv[i];
        else if (out.empty()) out = argv[i];
        else { std::fprintf(stderr, "unexpected argument %s\n", argv[i]); return 2; }
    }
    if (in.empty() || out.empty()) {
        std::fprintf(stderr, "usage: colossus_build IN.(ply|obj) OUT.cgeo [--up-z] [--check] [--max-triangles N] [--texture T.ppm] [--subdivide N] [--node NAME] [--keep-scale] [--normal-weight W]\n");
        return 2;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        auto since = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
        skeleton skin;
        mesh m = load_mesh(in, &skin, node);
        std::printf("read %s: %zu vertices, %zu triangles, %zu materials (%.1fs)\n", in.c_str(), m.positions.size(),
                    m.triangle_count(), m.materials.size(), since());
        if (m.textured() && !texture.empty()) m.materials[0].texture = texture;
        weld(m);
        if (m.triangle_count() == 0) throw std::runtime_error(in + ": no triangles");
        for (int k = 0; k < subdivisions; ++k) subdivide(m);
        if (subdivisions) std::printf("subdivided %d times: %zu triangles (%.1fs)\n", subdivisions, m.triangle_count(), since());
        skin.place = normalize_placement(m, up_z, keep_scale);
        std::printf("welded: %zu vertices, %zu triangles (%.1fs)\n", m.positions.size(), m.triangle_count(), since());
        // skinned: every animation at 16 times, and the rest pose, for measuring errors posed.
        std::vector<std::vector<float>> poses;
        if (m.skinned()) {
            std::vector<double> carried(skin.joints.size(), 0);
            for (size_t i = 0; i < m.skin_weights.size(); ++i) carried[m.skin_joints[i]] += m.skin_weights[i];
            skin.anchor = uint32_t(std::max_element(carried.begin(), carried.end()) - carried.begin());
            poses.emplace_back();
            pose_joints(skin, -1, 0, poses.back());
            for (size_t a = 0; a < skin.animations.size(); ++a)
                for (int k = 0; k < 16; ++k) {
                    poses.emplace_back();
                    pose_joints(skin, int(a), skin.animations[a].duration * k / 16, poses.back());
                }
            std::printf("skinned: %zu joints, %zu animations, errors measured in %zu poses\n", skin.joints.size(),
                        skin.animations.size(), poses.size());
        }
        const lod_mesh lod = build_lod(m, true, poses, normal_weight);
        geometry g = pack(lod);
        if (max_triangles) {
            g = trim(g, max_triangles);
            std::printf("trimmed to %zu triangles at the finest\n", g.leaf_triangles());
        }
        const paged_geometry paged = page(g);
        save_paged(paged, out);
        if (g.textured()) {
            // per material its colour (a ppm, or where it has none its flat colour), and where it
            // has them a normal map (its red and green as two textures) and a roughness map (its
            // green). an image used twice is written once.
            std::vector<texture_source> textures;
            std::vector<material_textures> materials;
            std::map<std::string, image> loaded;
            std::map<std::pair<std::string, int>, uint32_t> written;  // (file, channel or -1): index
            auto ppm_of = [](std::string file) {
                const size_t dot = file.find_last_of('.');
                if (file.compare(dot == std::string::npos ? file.size() : dot, std::string::npos, ".ppm") != 0) file = file.substr(0, dot) + ".ppm";
                return file;
            };
            auto add = [&](const std::string& file, int channel, uint8_t flags) -> uint32_t {
                const std::string ppm = ppm_of(file);
                auto key = std::make_pair(ppm, channel);
                if (auto it = written.find(key); it != written.end()) return it->second;
                if (!loaded.count(ppm)) {
                    try {
                        loaded[ppm] = load_ppm(ppm);
                    } catch (const std::exception& e) {
                        if (channel < 0) throw;
                        std::fprintf(stderr, "warning: %s: %s (no map)\n", ppm.c_str(), e.what());
                        return UINT32_MAX;
                    }
                }
                textures.push_back({loaded[ppm], channel >= 0, uint8_t(std::max(channel, 0)), flags});
                return written[key] = uint32_t(textures.size() - 1);
            };
            for (const material& mat : g.materials) {
                const uint8_t flags = uint8_t((mat.repeat ? 1 : 0) | (mat.double_sided ? 2 : 0));
                material_textures mt{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
                if (mat.texture.empty()) {
                    image im;
                    im.width = im.height = 4;
                    for (int i = 0; i < 16; ++i)
                        for (int c = 0; c < 3; ++c) {
                            const float l = std::clamp(mat.color[c], 0.0f, 1.0f);  // linear, to srgb
                            im.rgb.push_back(uint8_t(std::lround((l <= 0.0031308f ? 12.92f * l : 1.055f * std::pow(l, 1 / 2.4f) - 0.055f) * 255)));
                        }
                    textures.push_back({std::move(im), false, 0, flags});
                    mt[kind_colour] = uint32_t(textures.size() - 1);
                } else {
                    mt[kind_colour] = add(mat.texture, -1, flags);
                }
                if (!mat.normal_map.empty()) {
                    mt[kind_normal_x] = add(mat.normal_map, 0, flags);
                    mt[kind_normal_y] = mt[kind_normal_x] == UINT32_MAX ? UINT32_MAX : add(mat.normal_map, 1, flags);
                }
                if (!mat.roughness_map.empty()) mt[kind_roughness] = add(mat.roughness_map, 1, flags);
                materials.push_back(mt);
            }
            const std::string ctex = out.substr(0, out.find_last_of('.')) + ".ctex";
            save_textures(textures, materials, g.uv_min, g.uv_extent, ctex);
            const texture_info t = load_texture_info(ctex);
            std::printf("wrote %s: %zu materials, %zu textures, %u tiles, %.0f MB, coordinates over %.3g units (%.1fs)\n", ctex.c_str(),
                        t.materials.size(), t.textures.size(), t.tile_count(), t.tile_count() * double(tile_bytes) / 1048576.0, g.uv_extent, since());
        }
        if (m.skinned()) {
            const std::string cskn = out.substr(0, out.find_last_of('.')) + ".cskn";
            skin.pages = paged.skin;
            save_skeleton(skin, cskn);
            std::printf("wrote %s: %zu joints, %zu animations\n", cskn.c_str(), skin.joints.size(), skin.animations.size());
        }
        std::printf("wrote %s: %zu clusters over %zu levels, %zu triangles in all, %zu pages, %.0f MB (%.1fs)\n", out.c_str(),
                    g.clusters.size(), g.levels.size(), g.cluster_triangles.size(), paged.pages.size(),
                    paged.data_size / 1048576.0, since());
        if (check) {
            size_t bad = 0;
            for (const cut_report& r : check_cuts(g, 24)) {
                std::printf("  cut at %-10.3g %8zu clusters %10zu triangles  area %.4f  cracked edges %zu\n", r.threshold,
                            r.clusters, r.triangles, r.area_ratio, r.cracked_edges);
                bad += r.cracked_edges;
            }
            if (bad) {
                std::printf("FAIL: cracks in the hierarchy\n");
                return 1;
            }
            std::printf("every cut is crack-free\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
