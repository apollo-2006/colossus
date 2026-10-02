#pragma once
// Which pages live in the GPU's page pool: the CPU side of streaming.
//
// The GPU asks for pages: a cluster that is drawn but would rather be
// replaced by finer clusters, whose page is missing, requests that page,
// with its error on screen as the priority. Each frame the streamer reads
// the requests and issues loads for the most wanted pages, dependencies
// first, each into a slot of the pool (free, or taken from the least
// recently used page). Loader threads read the pages from disk; the render
// thread only issues loads and, frames later, publishes the ones that have
// finished: it hands back the copies to record and a new page table.
//
// One rule keeps every cut whole: a page may only be resident while its
// dependencies (the pages holding the coarser clusters its group was
// simplified into) are. So:
//   * loads are issued dependencies first, and published in the order they
//     were issued, so a page never goes live before its dependencies,
//     however the reads finish;
//   * a page counts as depending on its dependencies from the moment its
//     load is issued, and a page is evicted only when nothing resident or
//     loading depends on it.
// Then, wherever finer clusters are missing, the coarser ones that stand
// for them are there to be drawn instead, and the GPU's test (lod_test()
// in viewer/shaders/common.glsl) draws exactly one level along every path.
// The roots (page 0 of each model) never leave.
//
// A request also says how far from good enough its cluster is: its error
// on screen, against the threshold. Each level of the hierarchy roughly
// halves the error, so a cluster at four times the threshold will need
// two more levels; the streamer prefetches the finer pages below the one
// asked for, at half the priority a level, while that stays over the
// threshold, instead of waiting for each level to be drawn and ask in
// turn.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

struct stream_page {
    int fd;                // The model's file
    uint64_t file_offset;  // Where the page's bytes are in it
    uint32_t size;
    uint32_t dep_first, dep_count;  // Into the streamer's dependency list, global page numbers
    bool pinned;           // A root page: loaded first, never evicted
    uint32_t child_first = 0, child_count = 0;  // Finer pages below it, in the streamer's child list
};

class streamer {
public:
    static constexpr uint32_t none = UINT32_MAX;

    struct copy {
        uint64_t staging_offset, pool_offset, size;
    };
    struct stats_t {
        uint32_t resident = 0, slots = 0, loaded = 0, evicted = 0, waiting = 0, requested = 0, in_flight = 0;
        uint64_t bytes_loaded = 0;
    };

    // loader_threads 0 reads pages on the calling thread, as they are
    // issued, and publishes them in the same frame.
    streamer(std::vector<stream_page> pages, std::vector<uint32_t> deps, uint64_t pool_bytes, unsigned loader_threads = 2,
             std::vector<uint32_t> children = {})
        : pages_(std::move(pages)), deps_(std::move(deps)), children_(std::move(children)) {
        uint32_t largest = 16;
        for (const stream_page& p : pages_) largest = std::max(largest, p.size);
        slot_bytes_ = (largest + 4095) / 4096 * 4096;
        const uint64_t slots = pool_bytes / slot_bytes_;
        if (slots < 2) throw std::runtime_error("page pool too small for one page");
        slot_count_ = static_cast<uint32_t>(std::min<uint64_t>(slots, UINT32_MAX - 1));
        for (uint32_t s = slot_count_; s-- > 0;) free_.push_back(s);
        slot_of_.assign(pages_.size(), none);
        loading_.assign(pages_.size(), 0);
        last_used_.assign(pages_.size(), 0);
        dependents_.assign(pages_.size(), 0);
        table_.assign(pages_.size(), none);
        for (unsigned t = 0; t < loader_threads; ++t) loaders_.emplace_back([this] { load_loop(); });
    }

    ~streamer() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        queue_ready_.notify_all();
        for (std::thread& t : loaders_) t.join();
    }

    streamer(const streamer&) = delete;
    streamer& operator=(const streamer&) = delete;

    uint64_t slot_bytes() const { return slot_bytes_; }
    uint64_t pool_bytes() const { return uint64_t(slot_count_) * slot_bytes_; }
    // Per page: its first word in the pool, or none.
    const std::vector<uint32_t>& table() const { return table_; }
    // Published: in the table, for the GPU to draw from.
    bool resident(uint32_t page) const { return slot_of_[page] != none && !loading_[page]; }
    const stats_t& stats() const { return stats_; }

    // Pages the GPU drew from, as frame numbers per page.
    void note_used(const uint32_t* stamps) {
        for (size_t p = 0; p < pages_.size(); ++p) last_used_[p] = std::max(last_used_[p], stamps[p]);
    }

    // Publishes the loads that have finished (into `staging`, at most
    // staging_bytes this frame), then issues loads for what is asked for
    // (page, priority: higher first). Returns the copies from staging into
    // the pool to record before this frame's culling; table() is then this
    // frame's page table.
    std::vector<copy> service(uint32_t frame, std::vector<std::pair<uint32_t, float>> requests, uint8_t* staging,
                              uint64_t staging_bytes, float threshold = INFINITY) {
        frame_ = frame;
        stats_.loaded = stats_.evicted = 0;
        stats_.requested = static_cast<uint32_t>(requests.size());
        stats_.bytes_loaded = 0;
        std::vector<copy> copies;
        uint64_t used = 0;
        publish(staging, staging_bytes, copies, used);

        // Prefetch: the finer pages below each request, while their
        // predicted error stays over the threshold.
        if (!children_.empty()) {
            for (size_t r = 0; r < requests.size() && requests.size() < 8 * max_prefetch; ++r) {
                const auto [page, priority] = requests[r];
                if (page >= pages_.size() || priority * 0.5f <= threshold) continue;
                const stream_page& pg = pages_[page];
                for (uint32_t k = 0; k < pg.child_count; ++k) requests.push_back({children_[pg.child_first + k], priority * 0.5f});
            }
            // Each page once, at its highest priority.
            std::sort(requests.begin(), requests.end());
            size_t w = 0;
            for (size_t r = 0; r < requests.size(); ++r) {
                if (w > 0 && requests[w - 1].first == requests[r].first) requests[w - 1].second = std::max(requests[w - 1].second, requests[r].second);
                else requests[w++] = requests[r];
            }
            requests.resize(w);
        }
        // Roots first, on the first frames.
        for (uint32_t p = 0; p < pages_.size(); ++p)
            if (pages_[p].pinned && slot_of_[p] == none) requests.push_back({p, INFINITY});
        std::sort(requests.begin(), requests.end(), [](auto& a, auto& b) { return a.second > b.second; });
        candidates_ready_ = false;
        uint32_t waiting = 0;
        // Reads in flight are bounded too, so a burst of requests cannot
        // queue more than a few frames' worth of disk.
        const uint64_t flight_limit = 4 * staging_bytes;
        for (const auto& [page, priority] : requests) {
            if (page >= pages_.size() || slot_of_[page] != none) continue;
            // The page and every missing dependency below it, coarsest first.
            std::vector<uint32_t> chain;
            collect(page, chain);
            bool ok = true;
            for (uint32_t p : chain) {
                if (in_flight_bytes_ + pages_[p].size > flight_limit || !issue(p)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) ++waiting;
        }
        if (loaders_.empty()) publish(staging, staging_bytes, copies, used);
        stats_.waiting = waiting;
        stats_.in_flight = static_cast<uint32_t>(issued_.size());
        stats_.resident = slot_count_ - static_cast<uint32_t>(free_.size()) - stats_.in_flight;
        stats_.slots = slot_count_;
        return copies;
    }

private:
    struct job {
        uint32_t page;
        std::vector<uint8_t> data;
        std::atomic<bool> done{false};
    };

    static constexpr size_t max_prefetch = 4096;
    std::vector<stream_page> pages_;
    std::vector<uint32_t> deps_, children_;
    uint64_t slot_bytes_ = 0;
    uint32_t slot_count_ = 0;
    std::vector<uint32_t> free_;
    std::vector<uint32_t> slot_of_, last_used_, dependents_, table_;
    std::vector<uint8_t> loading_;
    std::vector<uint32_t> candidates_;  // Evictable pages, least recently used last
    bool candidates_ready_ = false;
    uint32_t frame_ = 0;
    stats_t stats_;

    std::deque<std::shared_ptr<job>> issued_;  // In issue order, until published
    uint64_t in_flight_bytes_ = 0;
    std::vector<std::thread> loaders_;
    std::mutex queue_mutex_;
    std::condition_variable queue_ready_;
    std::deque<std::shared_ptr<job>> queue_;  // Waiting for a loader thread
    bool stopping_ = false;

    static void read(job& j, const stream_page& pg) {
        j.data.resize(pg.size);
        size_t done = 0;
        while (done < pg.size) {
            const ssize_t n = pread(pg.fd, j.data.data() + done, pg.size - done, static_cast<off_t>(pg.file_offset + done));
            if (n <= 0) throw std::runtime_error("reading a page failed");
            done += static_cast<size_t>(n);
        }
        j.done.store(true, std::memory_order_release);
    }

    void load_loop() {
        for (;;) {
            std::shared_ptr<job> j;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_ready_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_) return;
                j = std::move(queue_.front());
                queue_.pop_front();
            }
            read(*j, pages_[j->page]);
        }
    }

    // Publishes finished loads in the order they were issued, stopping at
    // the first one still being read.
    void publish(uint8_t* staging, uint64_t staging_bytes, std::vector<copy>& copies, uint64_t& used) {
        while (!issued_.empty() && issued_.front()->done.load(std::memory_order_acquire)) {
            job& j = *issued_.front();
            const uint32_t size = pages_[j.page].size;
            if (used + size > staging_bytes) break;
            std::memcpy(staging + used, j.data.data(), size);
            const uint32_t s = slot_of_[j.page];
            copies.push_back({used, uint64_t(s) * slot_bytes_, size});
            used += (size + 15) / 16 * 16;
            loading_[j.page] = 0;
            table_[j.page] = static_cast<uint32_t>(uint64_t(s) * slot_bytes_ / 4);
            in_flight_bytes_ -= size;
            ++stats_.loaded;
            stats_.bytes_loaded += size;
            issued_.pop_front();
        }
    }

    // Lists the page and its missing dependencies, coarsest first. Its
    // dependencies already there are marked used this frame: making room
    // for the page must not evict what it is about to depend on.
    void collect(uint32_t page, std::vector<uint32_t>& chain) {
        if (slot_of_[page] != none) {
            last_used_[page] = frame_;
            return;
        }
        if (std::find(chain.begin(), chain.end(), page) != chain.end()) return;
        const stream_page& p = pages_[page];
        for (uint32_t k = 0; k < p.dep_count; ++k) collect(deps_[p.dep_first + k], chain);
        chain.push_back(page);
    }

    // A page may go if it is published, not a root, nothing resident or
    // loading depends on it, and nothing drew from it in the last two
    // frames.
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
            const stream_page& pg = pages_[p];
            for (uint32_t k = 0; k < pg.dep_count; ++k) --dependents_[deps_[pg.dep_first + k]];
            ++stats_.evicted;
            return s;
        }
        return none;
    }

    // Reserves a slot and starts the read; the page goes live when published.
    bool issue(uint32_t p) {
        const uint32_t s = take_slot();
        if (s == none) return false;
        const stream_page& pg = pages_[p];
        slot_of_[p] = s;
        loading_[p] = 1;
        last_used_[p] = frame_;  // Not to be evicted again this frame
        for (uint32_t k = 0; k < pg.dep_count; ++k) ++dependents_[deps_[pg.dep_first + k]];
        in_flight_bytes_ += pg.size;
        auto j = std::make_shared<job>();
        j->page = p;
        issued_.push_back(j);
        if (loaders_.empty()) {
            read(*j, pg);
        } else {
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                queue_.push_back(std::move(j));
            }
            queue_ready_.notify_one();
        }
        return true;
    }
};
