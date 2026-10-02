// Tests the streamer (viewer/streamer.hpp) on a real hierarchy, with no
// GPU: thousands of frames of random requests and random use, into a pool
// a tenth the size of the model, checking after every frame that the rule
// keeping cuts whole holds: every resident page's dependencies are
// resident. Then, with an ample pool and requests for everything, that
// every page can be loaded and the table points at distinct slots.
#include "dag.hpp"
#include "paged_file.hpp"
#include "../viewer/streamer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <map>
#include <random>
#include <set>
#include <unistd.h>

namespace {

int failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

mesh bumpy_sphere(int subdivisions) {
    mesh m;
    m.positions = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    m.indices = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
    for (int s = 0; s < subdivisions; ++s) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> mid;
        auto midpoint = [&](uint32_t a, uint32_t b) {
            const auto key = std::minmax(a, b);
            auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            m.positions.push_back(normalize(m.positions[a] + m.positions[b]));
            return mid[key] = static_cast<uint32_t>(m.positions.size() - 1);
        };
        std::vector<uint32_t> next;
        for (size_t t = 0; t < m.indices.size(); t += 3) {
            const uint32_t a = m.indices[t], b = m.indices[t + 1], c = m.indices[t + 2];
            const uint32_t ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            next.insert(next.end(), {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca});
        }
        m.indices = std::move(next);
    }
    for (vec3& p : m.positions) p = p * (1 + 0.03f * std::sin(9 * p.x) * std::sin(11 * p.y));
    return m;
}

// The rule: a resident page's dependencies are resident. (The root page
// is checked at the end: with loader threads it is in flight on the first
// frames, with nothing depending on it yet.)
size_t broken(const streamer& s, const paged_geometry& g) {
    size_t n = 0;
    for (uint32_t p = 0; p < g.pages.size(); ++p)
        if (s.resident(p))
            for (uint32_t k = 0; k < g.pages[p].dep_count; ++k) n += !s.resident(g.deps[g.pages[p].dep_first + k]);
    return n;
}

}  // namespace

int main() {
    mesh m = bumpy_sphere(7);
    normalize_placement(m, false);
    const paged_geometry g = page(pack(build_lod(m, false)));
    const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/colossus_streamer.cgeo";
    save_paged(g, path);
    const paged_geometry file = load_paged(path, false);
    const int fd = open(path.c_str(), O_RDONLY);
    std::vector<stream_page> pages;
    for (uint32_t p = 0; p < file.pages.size(); ++p)
        pages.push_back({fd, file.data_offset + file.pages[p].offset, file.pages[p].size, file.pages[p].dep_first,
                         file.pages[p].dep_count, p == 0});
    // Each page's children, for prefetching: the finer pages its clusters
    // stand for.
    std::vector<std::vector<uint32_t>> child_lists(file.pages.size());
    for (const gpu_cluster& c : file.clusters)
        if (c.creator != no_page) child_lists[c.group].push_back(c.creator);
    std::vector<uint32_t> children;
    for (uint32_t p = 0; p < file.pages.size(); ++p) {
        auto& ch = child_lists[p];
        std::sort(ch.begin(), ch.end());
        ch.erase(std::unique(ch.begin(), ch.end()), ch.end());
        pages[p].child_first = static_cast<uint32_t>(children.size());
        pages[p].child_count = static_cast<uint32_t>(ch.size());
        children.insert(children.end(), ch.begin(), ch.end());
    }
    std::printf("%zu pages, %.1f MB\n", file.pages.size(), file.data_size / 1048576.0);

    std::vector<uint8_t> staging(8 << 20);
    // Synchronous, then with two loader threads: pages published frames
    // after they were issued, in issue order.
    for (unsigned mode = 0; mode < 3; ++mode) {
        // Synchronous; loader threads; loader threads and prefetching.
        const unsigned threads = mode == 0 ? 0 : 2;
        const bool prefetch = mode == 2;
        std::printf("%u loader threads%s: random requests into a pool a tenth of the model\n", threads,
                    prefetch ? ", prefetching" : "");
        streamer s(pages, file.deps, file.data_size / 10, threads, prefetch ? children : std::vector<uint32_t>{});
        std::mt19937 rng(1);
        std::vector<uint32_t> stamps(file.pages.size(), 0);
        size_t worst = 0, loaded = 0, evicted = 0;
        for (uint32_t frame = 1; frame <= 3000; ++frame) {
            std::vector<std::pair<uint32_t, float>> requests;
            const int n = static_cast<int>(rng() % 64);
            for (int k = 0; k < n; ++k) requests.push_back({static_cast<uint32_t>(rng() % file.pages.size()), float(rng() % 1000)});
            // Some resident pages were drawn from, two frames ago.
            for (uint32_t p = 0; p < file.pages.size(); ++p)
                if (s.resident(p) && rng() % 4 == 0) stamps[p] = frame > 2 ? frame - 2 : 0;
            s.note_used(stamps.data());
            const auto copies = s.service(frame, requests, staging.data(), staging.size(), prefetch ? 1.0f : INFINITY);
            loaded += s.stats().loaded;
            evicted += s.stats().evicted;
            worst = std::max(worst, broken(s, file));
            // Copies land in distinct slots inside the pool, and the table
            // points there.
            std::set<uint64_t> slots;
            for (const auto& c : copies) {
                CHECK(c.pool_offset + c.size <= s.pool_bytes());
                CHECK(slots.insert(c.pool_offset).second);
            }
        }
        std::printf("  %zu loads, %zu evictions, broken dependencies at worst: %zu\n", loaded, evicted, worst);
        CHECK(worst == 0);
        CHECK(s.resident(0));
        CHECK(evicted > 50);  // The pool really was full and turning over

        std::printf("%u loader threads: everything, into a pool that holds it\n", threads);
        streamer all_in(pages, file.deps, file.data_size * 2, threads);
        std::vector<std::pair<uint32_t, float>> all;
        for (uint32_t p = 0; p < file.pages.size(); ++p) all.push_back({p, 1});
        size_t resident = 0;
        for (uint32_t frame = 1; frame <= 2000 && resident < file.pages.size(); ++frame) {
            all_in.service(frame, all, staging.data(), staging.size());
            CHECK(broken(all_in, file) == 0);
            resident = 0;
            for (uint32_t p = 0; p < file.pages.size(); ++p) resident += all_in.resident(p);
            if (threads) usleep(200);
        }
        std::set<uint32_t> offsets;
        for (uint32_t p = 0; p < file.pages.size(); ++p)
            if (all_in.resident(p)) CHECK(offsets.insert(all_in.table()[p]).second);
        std::printf("  %zu of %zu pages resident\n", resident, file.pages.size());
        CHECK(resident == file.pages.size());
    }
    close(fd);
    if (failures) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
}
