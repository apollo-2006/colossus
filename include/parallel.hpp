#pragma once
// parallel_for over [0, n): one worker per hardware thread, each taking the
// next index from a shared counter, so uneven items balance themselves.
#include <atomic>
#include <thread>
#include <vector>

template <class Body>
void parallel_for(size_t n, Body&& body) {
    const size_t workers = std::min<size_t>(n, std::max(1u, std::thread::hardware_concurrency()));
    if (workers <= 1) {
        for (size_t i = 0; i < n; ++i) body(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t w = 0; w < workers; ++w)
        threads.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n;) body(i);
        });
    for (std::thread& t : threads) t.join();
}
