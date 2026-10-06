#pragma once
// the viewer's command line.
#include "gpu_types.hpp"
#include "math.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace viewer {

struct options {
    std::vector<std::string> models;
    int grid = 1;
    float spacing = 1.4f;
    int width = 1600, height = 900;
    bool headless = false, validate = false, vsync = true;
    int frames = 0;  // headless: frames drawn and timed
    std::string screenshot;
    std::string record;  // headless: every frame after the warmup as DIR/frame_NNNNN.png
    std::string path;    // headless: a camera path (camera.hpp) followed from the first recorded frame
    float threshold = 1.0f;
    uint32_t mode = 0;
    bool camera_set = false;
    vec3 eye;
    float yaw = 0, pitch = 0;
    bool wireframe = false;
    uint32_t disable = 0;  // flags turned off on the command line
    bool cull_only = false;
    float sw_pixels = 32;
    uint64_t pool_mb = 0;      // page pool: 0 for the scene's (`pool` in a scene file) or 1024
    bool vsm = true;           // virtual shadow maps, not ray queries (--shadows rt)
    uint32_t vsm_side = 48;    // physical shadow pages a side: 48 is 2304 pages, 288 mb (two layers)
    uint64_t upload_mb = 64;   // pages loaded per frame, at most
    int warmup = 0;            // headless: frames before timing
    float fly = 0;             // headless: camera moves this far forward per frame, turning slowly
    unsigned loader_threads = 8;  // 0: read pages on the render thread. mostly waiting on disk: nvme wants queue depth
    bool cold = false;         // evict the models from the os file cache first
    bool mixed_materials = true;
    int material = -1;  // --materials NAME: every instance one material (shade.comp's table)
    float moving = 0;  // share of moving instances (animate() in common.glsl)
    float deforming = 0;  // share of deforming instances (deform() in common.glsl)
    bool full_res_shadows = false;
    bool prefetch = true;
    bool merge_reads = true;   // pages close in the file share a read
    bool gi_rt = false;        // --gi rt: bounce light traced in world space
    std::string animation;     // skinned models: play only this animation (default: each at random)
    std::string scene;         // --scene FILE: models and their placements (scene.hpp) in place of --model and --grid
};

// command line to options; exits with usage on a bad one.
options parse(int argc, char** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") o.models.push_back(next());
        else if (a == "--grid") o.grid = std::stoi(next());
        else if (a == "--spacing") o.spacing = std::stof(next());
        else if (a == "--width") o.width = std::stoi(next());
        else if (a == "--height") o.height = std::stoi(next());
        else if (a == "--headless") o.headless = true;
        else if (a == "--frames") o.frames = std::stoi(next());
        else if (a == "--screenshot") o.screenshot = next();
        else if (a == "--record") o.record = next();
        else if (a == "--path") o.path = next();
        else if (a == "--threshold") o.threshold = std::stof(next());
        else if (a == "--mode") o.mode = std::min<uint32_t>(std::stoul(next()), mode_count - 1);
        else if (a == "--validate") o.validate = true;
        else if (a == "--no-vsync") o.vsync = false;
        else if (a == "--wireframe") o.wireframe = true;
        else if (a == "--no-occlusion") o.disable |= flag_occlusion;
        else if (a == "--no-cone") o.disable |= flag_cone_culling;
        else if (a == "--no-sw") o.disable |= flag_software_raster;
        else if (a == "--no-shadows") o.disable |= flag_shadows;
        else if (a == "--shadows") {
            const std::string kind = next();
            if (kind != "rt" && kind != "vsm") throw std::runtime_error("--shadows takes rt or vsm");
            o.vsm = kind == "vsm";
        } else if (a == "--vsm-pages") {
            o.vsm_side = static_cast<uint32_t>(std::stoul(next()));
            // the evictable list holds half a slot count each of stale and recent pages.
            if (o.vsm_side == 0 || o.vsm_side * o.vsm_side > vsm_slots / 2) throw std::runtime_error("--vsm-pages takes 1 to 84");
        }
        else if (a == "--no-taa") o.disable |= flag_taa;
        else if (a == "--no-ao") o.disable |= flag_ao;
        else if (a == "--no-bounce") o.disable |= flag_bounce;
        else if (a == "--gi") {
            const std::string g = next();
            if (g != "rt" && g != "screen") throw std::runtime_error("--gi takes rt or screen");
            o.gi_rt = g == "rt";
        }
        else if (a == "--hard-shadows") o.disable |= flag_soft_shadows;
        else if (a == "--full-res-shadows") o.full_res_shadows = true;
        else if (a == "--materials") {
            const std::string m = next();
            const char* names[] = {"plain", "marble", "sandstone", "bronze", "gold", "granite"};
            const auto it = std::find(std::begin(names), std::end(names), m);
            if (m != "mixed" && it == std::end(names))
                throw std::runtime_error("--materials takes mixed, plain, marble, sandstone, bronze, gold or granite");
            o.mixed_materials = m != "plain";
            if (it != std::end(names) && m != "plain") o.material = int(it - std::begin(names));
        }
        else if (a == "--moving") o.moving = std::stof(next());
        else if (a == "--deforming") o.deforming = std::stof(next());
        else if (a == "--animation") o.animation = next();
        else if (a == "--scene") o.scene = next();
        else if (a == "--pool-mb") o.pool_mb = std::clamp<uint64_t>(std::stoull(next()), 16, 4095);
        else if (a == "--upload-mb") o.upload_mb = std::clamp<uint64_t>(std::stoull(next()), 1, 1024);
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--fly") o.fly = std::stof(next());
        else if (a == "--sync-loads") o.loader_threads = 0;
        else if (a == "--loader-threads") o.loader_threads = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--cold") o.cold = true;
        else if (a == "--no-prefetch") o.prefetch = false;
        else if (a == "--no-merge-reads") o.merge_reads = false;
        else if (a == "--sw-pixels") o.sw_pixels = std::clamp(std::stof(next()), 0.0f, 64.0f);  // sw_raster.comp's 32-bit math holds to 64
        else if (a == "--cull-only") o.cull_only = true;
        else if (a == "--camera") {
            float x, y, z, yaw, pitch;
            if (std::sscanf(next().c_str(), "%f,%f,%f,%f,%f", &x, &y, &z, &yaw, &pitch) != 5)
                throw std::runtime_error("--camera wants x,y,z,yaw,pitch");
            o.eye = {x, y, z};
            o.yaw = yaw;
            o.pitch = pitch;
            o.camera_set = true;
        } else throw std::runtime_error("unknown argument " + a);
    }
    if (o.models.empty() && o.scene.empty())
        throw std::runtime_error("usage: colossus --model FILE.cgeo [--model ...] [--grid N] | --scene FILE  [--headless --frames N --screenshot out.png]");
    if (o.headless && o.frames <= 0) o.frames = 1;
    return o;
}

}  // namespace viewer
