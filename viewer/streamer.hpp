#pragma once
// which pages live in the gpu's page pool: streaming's cpu side.
//
// the gpu asks: a drawn cluster that would rather be its finer clusters, whose
// page is missing, requests it, with its error on screen as priority. each
// frame the streamer issues loads for the most wanted pages, dependencies
// first, into free or least recently used slots. loader threads read; the
// render thread only issues and, frames later, publishes finished loads: copies
// to record and a new page table.
//
// one rule keeps cuts whole: a page is resident only while its dependencies
// (pages of the coarser clusters its group simplified into) are. so:
//   * loads are issued dependencies first and published in issue order, so a
//     page never goes live before them;
//   * a page depends on its dependencies from the moment its load is issued,
//     and is evicted only when nothing resident or loading depends on it.
// wherever finer clusters are missing, their stand-ins are there, and
// lod_test() (viewer/shaders/common.glsl) draws one level per path. roots (each
// model's page 0) never leave.
//
// each level roughly halves the error, so a cluster at four times the threshold
// needs two more levels: the streamer prefetches the pages below a request at
// half the priority a level, while the predicted error stays over the
// threshold.
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
    int fd;                // the model's file
    uint64_t file_offset;  // where the page's bytes are
    uint32_t size;
    uint32_t dep_first, dep_count;  // into the dependency list, global page numbers
    bool pinned;           // root: loaded first, never evicted
    uint32_t child_first = 0, child_count = 0;  // finer pages below it, in the child list
    float error = 0;  // its clusters' parent error, model units: what a request for it measured
};

class streamer {
public:
    static constexpr uint32_t none = UINT32_MAX;

    struct copy {
        uint64_t staging_offset, pool_offset, size;
    };
    struct stats_t {
        uint32_t resident = 0, slots = 0, loaded = 0, evicted = 0, waiting = 0, requested = 0, in_flight = 0;
        uint32_t reads = 0;  // reads issued: pages close together in the file share one
        uint64_t bytes_loaded = 0;
    };

    // loader_threads 0 reads on the calling thread as issued, publishing the
    // same frame.
    streamer(std::vector<stream_page> pages, std::vector<uint32_t> deps, uint64_t pool_bytes, unsigned loader_threads = 2,
             std::vector<uint32_t> children = {}, bool merge_reads = true)
        : pages_(std::move(pages)), deps_(std::move(deps)), children_(std::move(children)), merge_reads_(merge_reads) {
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
    // per page: its first word in the pool, or none.
    const std::vector<uint32_t>& table() const { return table_; }
    // published: in the table, drawable.
    bool resident(uint32_t page) const { return slot_of_[page] != none && !loading_[page]; }
    const stats_t& stats() const { return stats_; }

    // pages the gpu drew from, as frame numbers.
    void note_used(const uint32_t* stamps) {
        for (size_t p = 0; p < pages_.size(); ++p) last_used_[p] = std::max(last_used_[p], stamps[p]);
    }

    // publishes finished loads (into `staging`, at most staging_bytes), then
    // issues loads for requests (page, priority: higher first). returns the
    // copies from staging to the pool to record before culling; table() is then
    // this frame's page table. throws when a page's read failed.
    std::vector<copy> service(uint32_t frame, std::vector<std::pair<uint32_t, float>> requests, uint8_t* staging,
                              uint64_t staging_bytes, float threshold = INFINITY) {
        frame_ = frame;
        stats_.loaded = stats_.evicted = stats_.reads = 0;
        stats_.requested = static_cast<uint32_t>(requests.size());
        stats_.bytes_loaded = 0;
        std::vector<copy> copies;
        uint64_t used = 0;
        publish(staging, staging_bytes, copies, used);

        // prefetch: finer pages below each request while their predicted error stays over the
        // threshold. a request's priority is its page's error on screen, so priority / error is
        // the scale, and a child's prediction is its own error at that scale.
        if (!children_.empty()) {
            for (size_t r = 0; r < requests.size() && requests.size() < 8 * max_prefetch; ++r) {
                const auto [page, priority] = requests[r];
                if (page >= pages_.size()) continue;
                const stream_page& pg = pages_[page];
                const bool known = pg.error > 0 && std::isfinite(pg.error);
                for (uint32_t k = 0; k < pg.child_count; ++k) {
                    const uint32_t child = children_[pg.child_first + k];
                    const float predicted = known ? priority * (pages_[child].error / pg.error) : priority * 0.5f;
                    if (predicted > threshold) requests.push_back({child, predicted});
                }
            }
            // each page once, at its highest priority.
            std::sort(requests.begin(), requests.end());
            size_t w = 0;
            for (size_t r = 0; r < requests.size(); ++r) {
                if (w > 0 && requests[w - 1].first == requests[r].first) requests[w - 1].second = std::max(requests[w - 1].second, requests[r].second);
                else requests[w++] = requests[r];
            }
            requests.resize(w);
        }
        // roots first, on the first frames.
        for (uint32_t p = 0; p < pages_.size(); ++p)
            if (pages_[p].pinned && slot_of_[p] == none) requests.push_back({p, INFINITY});
        std::sort(requests.begin(), requests.end(), [](auto& a, auto& b) { return a.second > b.second; });
        candidates_ready_ = false;
        uint32_t waiting = 0;
        // reads in flight are bounded too: a burst cannot queue more than a few
        // frames of disk.
        const uint64_t flight_limit = 4 * staging_bytes;
        for (const auto& [page, priority] : requests) {
            if (page >= pages_.size() || slot_of_[page] != none) continue;
            // the page and every missing dependency, coarsest first.
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
        flush_batch();
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
        bool failed = false;  // the read did: published as an error on the render thread
        std::atomic<bool> done{false};
    };
    // one read covering pages close together in a file; each job gets its slice.
    struct run {
        int fd = -1;
        uint64_t offset = 0, size = 0;
        std::vector<std::shared_ptr<job>> jobs;
    };
    // pages issued in one frame merge into a read when the gap between them is at most this
    // (read and dropped), up to max_run bytes: the file keeps each level's groups together,
    // and a frame's requests are mostly siblings.
    static constexpr uint64_t max_gap = 64 << 10, max_run = 4 << 20;
    std::vector<std::shared_ptr<job>> batch_;  // issued this frame, not yet read

    static constexpr size_t max_prefetch = 4096;
    std::vector<stream_page> pages_;
    std::vector<uint32_t> deps_, children_;
    bool merge_reads_ = true;  // else a read per page
    uint64_t slot_bytes_ = 0;
    uint32_t slot_count_ = 0;
    std::vector<uint32_t> free_;
    std::vector<uint32_t> slot_of_, last_used_, dependents_, table_;
    std::vector<uint8_t> loading_;
    std::vector<uint32_t> candidates_;  // evictable, least recently used last
    bool candidates_ready_ = false;
    uint32_t frame_ = 0;
    stats_t stats_;

    std::deque<std::shared_ptr<job>> issued_;  // in issue order, until published
    uint64_t in_flight_bytes_ = 0;
    std::vector<std::thread> loaders_;
    std::mutex queue_mutex_;
    std::condition_variable queue_ready_;
    std::deque<run> queue_;  // waiting for a loader thread
    bool stopping_ = false;

    void read(const run& r) {
        std::vector<uint8_t> buffer(r.size);
        size_t done = 0;
        while (done < r.size) {
            const ssize_t n = pread(r.fd, buffer.data() + done, r.size - done, static_cast<off_t>(r.offset + done));
            if (n <= 0) {
                // not thrown here: on a loader thread that would terminate.
                for (const auto& j : r.jobs) {
                    j->failed = true;
                    j->done.store(true, std::memory_order_release);
                }
                return;
            }
            done += static_cast<size_t>(n);
        }
        for (const auto& j : r.jobs) {
            const stream_page& pg = pages_[j->page];
            const uint8_t* from = buffer.data() + (pg.file_offset - r.offset);
            j->data.assign(from, from + pg.size);
            j->done.store(true, std::memory_order_release);
        }
    }

    // this frame's issued pages, sorted by place in the file and merged into runs.
    void flush_batch() {
        std::sort(batch_.begin(), batch_.end(), [&](const auto& a, const auto& b) {
            const stream_page &x = pages_[a->page], &y = pages_[b->page];
            return x.fd != y.fd ? x.fd < y.fd : x.file_offset < y.file_offset;
        });
        std::vector<run> runs;
        for (auto& j : batch_) {
            const stream_page& pg = pages_[j->page];
            if (!runs.empty() && merge_reads_) {
                run& r = runs.back();
                const uint64_t end = r.offset + r.size;
                if (r.fd == pg.fd && pg.file_offset >= end && pg.file_offset - end <= max_gap &&
                    pg.file_offset + pg.size - r.offset <= max_run) {
                    r.size = pg.file_offset + pg.size - r.offset;
                    r.jobs.push_back(std::move(j));
                    continue;
                }
            }
            runs.push_back({pg.fd, pg.file_offset, pg.size, {std::move(j)}});
        }
        batch_.clear();
        stats_.reads += static_cast<uint32_t>(runs.size());
        if (loaders_.empty()) {
            for (const run& r : runs) read(r);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for (run& r : runs) queue_.push_back(std::move(r));
        }
        queue_ready_.notify_all();
    }

    void load_loop() {
        for (;;) {
            run r;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_ready_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_) return;
                r = std::move(queue_.front());
                queue_.pop_front();
            }
            read(r);
        }
    }

    // publishes finished loads in issue order, stopping at the first still
    // reading. throws at a failed read.
    void publish(uint8_t* staging, uint64_t staging_bytes, std::vector<copy>& copies, uint64_t& used) {
        while (!issued_.empty() && issued_.front()->done.load(std::memory_order_acquire)) {
            job& j = *issued_.front();
            if (j.failed) throw std::runtime_error("reading a page failed");
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

    // lists the page and its missing dependencies, coarsest first. dependencies
    // present are marked used: making room must not evict them.
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

    // evictable: published, not a root, nothing resident or loading depends on
    // it, unused for two frames.
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
            if (!evictable(p)) continue;  // something came to depend on it this frame
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

    // reserves a slot and starts the read; live when published.
    bool issue(uint32_t p) {
        const uint32_t s = take_slot();
        if (s == none) return false;
        const stream_page& pg = pages_[p];
        slot_of_[p] = s;
        loading_[p] = 1;
        last_used_[p] = frame_;  // not to be evicted again this frame
        for (uint32_t k = 0; k < pg.dep_count; ++k) ++dependents_[deps_[pg.dep_first + k]];
        in_flight_bytes_ += pg.size;
        auto j = std::make_shared<job>();
        j->page = p;
        issued_.push_back(j);
        batch_.push_back(std::move(j));  // read by flush_batch()
        return true;
    }
};
