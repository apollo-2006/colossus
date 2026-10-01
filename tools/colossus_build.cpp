// colossus_build: reads a PLY or OBJ model, builds its cluster LOD hierarchy and
// writes it as a .cgeo file for the viewer.
//
//   colossus_build models/lucy.ply models/lucy.cgeo --up-z
//   colossus_build in.obj out.cgeo --check     also check every cut for cracks
//   colossus_build in.ply out.cgeo --max-triangles 500000
//                                          keep only the levels from the finest
//                                          cut within 500k triangles up
#include "geometry_file.hpp"
#include "lod_check.hpp"
#include "paged_file.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

int main(int argc, char** argv) {
    std::string in, out;
    bool up_z = false, check = false;
    size_t max_triangles = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--up-z")) up_z = true;
        else if (!std::strcmp(argv[i], "--max-triangles") && i + 1 < argc) max_triangles = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--check")) check = true;
        else if (in.empty()) in = argv[i];
        else if (out.empty()) out = argv[i];
        else { std::fprintf(stderr, "unexpected argument %s\n", argv[i]); return 2; }
    }
    if (in.empty() || out.empty()) {
        std::fprintf(stderr, "usage: colossus_build IN.(ply|obj) OUT.cgeo [--up-z] [--check] [--max-triangles N]\n");
        return 2;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        auto since = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
        mesh m = load_mesh(in);
        std::printf("read %s: %zu vertices, %zu triangles (%.1fs)\n", in.c_str(), m.positions.size(), m.triangle_count(), since());
        weld(m);
        normalize_placement(m, up_z);
        std::printf("welded: %zu vertices, %zu triangles (%.1fs)\n", m.positions.size(), m.triangle_count(), since());
        const lod_mesh lod = build_lod(m, true);
        geometry g = pack(lod);
        if (max_triangles) {
            g = trim(g, max_triangles);
            std::printf("trimmed to %zu triangles at the finest\n", g.leaf_triangles());
        }
        const paged_geometry paged = page(g);
        save_paged(paged, out);
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
