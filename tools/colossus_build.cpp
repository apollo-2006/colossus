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
//
// a textured obj also writes out.ctex beside out.cgeo (include/texture_file.hpp). the
// texture is read as a binary ppm: one in another format is looked for under the same name
// ending .ppm (ffmpeg -i t.jpg t.ppm makes one). a skinned gltf also writes out.cskn: its
// skeleton, animations and per page skin bounds (include/skeleton.hpp).
#include "geometry_file.hpp"
#include "lod_check.hpp"
#include "paged_file.hpp"
#include "skeleton.hpp"
#include "texture_file.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    std::string in, out, texture;
    bool up_z = false, check = false;
    size_t max_triangles = 0;
    int subdivisions = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--up-z")) up_z = true;
        else if (!std::strcmp(argv[i], "--max-triangles") && i + 1 < argc) max_triangles = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--texture") && i + 1 < argc) texture = argv[++i];
        else if (!std::strcmp(argv[i], "--subdivide") && i + 1 < argc) subdivisions = std::atoi(argv[++i]);
        else if (in.empty()) in = argv[i];
        else if (out.empty()) out = argv[i];
        else { std::fprintf(stderr, "unexpected argument %s\n", argv[i]); return 2; }
    }
    if (in.empty() || out.empty()) {
        std::fprintf(stderr, "usage: colossus_build IN.(ply|obj) OUT.cgeo [--up-z] [--check] [--max-triangles N] [--texture T.ppm] [--subdivide N]\n");
        return 2;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        auto since = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
        skeleton skin;
        mesh m = load_mesh(in, &skin);
        std::printf("read %s: %zu vertices, %zu triangles (%.1fs)\n", in.c_str(), m.positions.size(), m.triangle_count(), since());
        if (m.textured()) {
            if (texture.empty()) texture = m.texture;
            if (texture.empty()) throw std::runtime_error(in + ": texture coordinates but no texture: give --texture");
            const size_t dot = texture.find_last_of('.');
            if (texture.compare(dot == std::string::npos ? texture.size() : dot, std::string::npos, ".ppm") != 0)
                texture = texture.substr(0, dot) + ".ppm";
            const std::string ctex = out.substr(0, out.find_last_of('.')) + ".ctex";
            const image im = load_ppm(texture);
            save_texture(im, ctex);
            const texture_info t = load_texture_info(ctex);
            std::printf("wrote %s: %u x %u, %zu levels, %u tiles, %.0f MB (%.1fs)\n", ctex.c_str(), im.width, im.height,
                        t.levels.size(), t.tile_count(), t.tile_count() * double(tile_bytes) / 1048576.0, since());
        }
        weld(m);
        if (m.triangle_count() == 0) throw std::runtime_error(in + ": no triangles");
        for (int k = 0; k < subdivisions; ++k) subdivide(m);
        if (subdivisions) std::printf("subdivided %d times: %zu triangles (%.1fs)\n", subdivisions, m.triangle_count(), since());
        skin.place = normalize_placement(m, up_z);
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
        const lod_mesh lod = build_lod(m, true, poses);
        geometry g = pack(lod);
        if (max_triangles) {
            g = trim(g, max_triangles);
            std::printf("trimmed to %zu triangles at the finest\n", g.leaf_triangles());
        }
        const paged_geometry paged = page(g);
        save_paged(paged, out);
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
