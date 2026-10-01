#pragma once
// Which pages live in the GPU's page pool: the CPU side of streaming.
//
// The GPU asks for pages: a cluster that is drawn but would rather be
// replaced by finer clusters, whose page is missing, requests that page,
// with its error on screen as the priority. Each frame the streamer reads
// the requests, loads the most wanted pages from disk into free slots of
// the pool (evicting the least recently used pages when it is full), and
// hands back the copies to record and a new page table.
//
// One rule keeps every cut whole: a page may only be resident while its
// dependencies (the pages holding the coarser clusters its group was
// simplified into) are. So a page is loaded only after its dependencies,
// and evicted only when no resident page depends on it. Then, wherever
// finer clusters are missing, the coarser ones that stand for them are
// there to be drawn instead, and the GPU's test (see lod_test() in
// viewer/shaders/common.glsl) draws exactly one level along every path.
// The roots (page 0 of each model) never leave.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

struct stream_page {
    int fd;                // The model's file
    uint64_t file_offset;  // Where the page's bytes are in it
    uint32_t size;
    uint32_t dep_first, dep_count;  // Into the streamer's dependency list, global page numbers
    bool pinned;           // A root page: loaded first, never evicted
};

class streamer {
public:
    static constexpr uint32_t none = UINT32_MAX;

    struct copy {
        uint64_t staging_offset, pool_offset, size;
    };
    struct stats_t {
        uint32_t resident = 0, slots = 0, loaded = 0, evicted = 0, waiting = 0, requested = 0;
        uint64_t bytes_loaded = 0;
    };

    streamer(std::vector<stream_page> pages, std::vector<uint32_t> deps, uint64_t pool_bytes)
        : pages_(std::move(pages)), deps_(std::move(deps)) {
        uint32_t largest = 16;
        for (const stream_page& p : pages_) largest = std::max(largest, p.size);
        slot_bytes_ = (largest + 4095) / 4096 * 4096;
        const uint64_t slots = pool_bytes / slot_bytes_;
        if (slots < 2) throw std::runtime_error("page pool too small for one page");
        slot_count_ = static_cast<uint32_t>(std::min<uint64_t>(slots, UINT32_MAX - 1));
        for (uint32_t s = slot_count_; s-- > 0;) free_.push_back(s);
        slot_of_.assign(pages_.size(), none);
        page_in_slot_.assign(slot_count_, none);
        last_used_.assign(pages_.size(), 0);
        dependents_.assign(pages_.size(), 0);
        table_.assign(pages_.size(), none);
    }

    uint64_t slot_bytes() const { return slot_bytes_; }
    uint64_t pool_bytes() const { return uint64_t(slot_count_) * slot_bytes_; }
    // Per page: its first word in the pool, or none.
    const std::vector<uint32_t>& table() const { return table_; }
    bool resident(uint32_t page) const { return slot_of_[page] != none; }
    const stats_t& stats() const { return stats_; }

    // Pages the GPU drew from, as frame numbers per page.
    void note_used(const uint32_t* stamps) {
        for (size_t p = 0; p < pages_.size(); ++p) last_used_[p] = std::max(last_used_[p], stamps[p]);
    }

    // Loads what is asked for (page, priority: higher first), dependencies
    // first, into `staging` (at most staging_bytes this frame). Returns the
    // copies from staging into the pool to record before this frame's
    // culling; table() is then this frame's page table.
    std::vector<copy> service(uint32_t frame, std::vector<std::pair<uint32_t, float>> requests, uint8_t* staging,
                              uint64_t staging_bytes) {
        frame_ = frame;
        stats_.loaded = stats_.evicted = 0;
        stats_.requested = static_cast<uint32_t>(requests.size());
        stats_.bytes_loaded = 0;
        std::vector<copy> copies;
        uint64_t used = 0;
        // Roots first, on the first frames.
        for (uint32_t p = 0; p < pages_.size(); ++p)
            if (pages_[p].pinned && !resident(p)) requests.push_back({p, INFINITY});
        std::sort(requests.begin(), requests.end(), [](auto& a, auto& b) { return a.second > b.second; });
        candidates_ready_ = false;
        uint32_t waiting = 0;
        for (const auto& [page, priority] : requests) {
            if (page >= pages_.size() || resident(page)) continue;
            // The page and every missing dependency below it, coarsest first.
            std::vector<uint32_t> chain;
            collect(page, chain);
            bool ok = true;
            for (uint32_t p : chain) {
                if (resident(p)) continue;
                if (used + pages_[p].size > staging_bytes || !load(p, staging + used, copies, used)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) ++waiting;
        }
        stats_.waiting = waiting;
        stats_.resident = slot_count_ - static_cast<uint32_t>(free_.size());
        stats_.slots = slot_count_;
        return copies;
    }

private:
    std::vector<stream_page> pages_;
    std::vector<uint32_t> deps_;
    uint64_t slot_bytes_ = 0;
    uint32_t slot_count_ = 0;
    std::vector<uint32_t> free_;
    std::vector<uint32_t> slot_of_, page_in_slot_, last_used_, dependents_, table_;
    std::vector<uint32_t> candidates_;  // Evictable pages, least recently used last
    bool candidates_ready_ = false;
    uint32_t frame_ = 0;
    stats_t stats_;

    // Lists the page and its missing dependencies, coarsest first. Its
    // resident dependencies are marked used this frame: making room for
    // the page must not evict what it is about to depend on.
    void collect(uint32_t page, std::vector<uint32_t>& chain) {
        if (resident(page)) {
            last_used_[page] = frame_;
            return;
        }
        if (std::find(chain.begin(), chain.end(), page) != chain.end()) return;
        const stream_page& p = pages_[page];
        for (uint32_t k = 0; k < p.dep_count; ++k) collect(deps_[p.dep_first + k], chain);
        chain.push_back(page);
    }

    // A page may go if it is not a root, nothing resident depends on it,
    // and nothing drew from it in the last two frames.
    bool evictable(uint32_t p) const {
        return resident(p) && !pages_[p].pinned && dependents_[p] == 0 && last_used_[p] + 2 <= frame_;
    }

    uint32_t take_slot() {
        if (!free_.empty()) {
            const uint32_t s = free_.back();
            free_.pop_back();
            return s;
        }
        if (!candidates_ready_) {
            candidates_.clear();
            for (uint32_t p = 0; p < pages_.size(); ++p)
                if (evictable(p)) candidates_.push_back(p);
            std::sort(candidates_.begin(), candidates_.end(), [&](uint32_t a, uint32_t b) { return last_used_[a] > last_used_[b]; });
            candidates_ready_ = true;
        }
        while (!candidates_.empty()) {
            const uint32_t p = candidates_.back();
            candidates_.pop_back();
            if (!evictable(p)) continue;  // Something came to depend on it this frame
            const uint32_t s = slot_of_[p];
            slot_of_[p] = none;
            table_[p] = none;
            page_in_slot_[s] = none;
            const stream_page& pg = pages_[p];
            for (uint32_t k = 0; k < pg.dep_count; ++k) --dependents_[deps_[pg.dep_first + k]];
            ++stats_.evicted;
            return s;
        }
        return none;
    }

    bool load(uint32_t p, uint8_t* dst, std::vector<copy>& copies, uint64_t& used) {
        const uint32_t s = take_slot();
        if (s == none) return false;
        const stream_page& pg = pages_[p];
        size_t done = 0;
        while (done < pg.size) {
            const ssize_t n = pread(pg.fd, dst + done, pg.size - done, static_cast<off_t>(pg.file_offset + done));
            if (n <= 0) throw std::runtime_error("reading a page failed");
            done += static_cast<size_t>(n);
        }
        copies.push_back({used, uint64_t(s) * slot_bytes_, pg.size});
        used += (pg.size + 15) / 16 * 16;
        slot_of_[p] = s;
        page_in_slot_[s] = p;
        table_[p] = static_cast<uint32_t>(uint64_t(s) * slot_bytes_ / 4);
        last_used_[p] = frame_;  // Not to be evicted again this frame
        for (uint32_t k = 0; k < pg.dep_count; ++k) ++dependents_[deps_[pg.dep_first + k]];
        ++stats_.loaded;
        stats_.bytes_loaded += pg.size;
        return true;
    }
};
