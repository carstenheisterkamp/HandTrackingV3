#pragma once

#include <atomic>
#include <array>
#include <optional>
#include <cassert>
#include <cstddef>

namespace core {

/**
 * Single-Producer-Single-Consumer (SPSC) Lock-free Ringbuffer.
 * Optimized for low-latency pointer passing between threads.
 *
 * Uses std::atomic with acquire/release semantics to ensure memory visibility
 * without full memory barriers or mutexes.
 *
 * @tparam T Type of the element (usually a shared_ptr or unique_ptr)
 * @tparam Capacity Size of the ringbuffer
 */
template<typename T, size_t Capacity>
class SpscQueue {
public:
    SpscQueue() : head_(0), tail_(0) {}

    // Non-copyable
    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    /**
     * Pushes an item into the queue.
     * Returns false if the queue is full.
     * Thread-safety: Only safe to call from the Producer thread.
     */
    bool try_push(T item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) % Capacity;

        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }

        buffer_[current_tail] = std::move(item);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    /**
     * Pops an item from the queue.
     * Returns std::nullopt if empty.
     * Thread-safety: Only safe to call from the Consumer thread.
     */
    std::optional<T> try_pop() {
        const size_t current_head = head_.load(std::memory_order_relaxed);

        if (current_head == tail_.load(std::memory_order_acquire)) {
            return std::nullopt; // Queue is empty
        }

        T item = std::move(buffer_[current_head]);
        head_.store((current_head + 1) % Capacity, std::memory_order_release);
        return item;
    }

    /**
     * Reads an item from the queue into the provided reference.
     * Returns true if successful, false if queue is empty.
     * Thread-safety: Only safe to call from the Consumer thread.
     */
    bool pop_front(T& item) {
        auto val = try_pop();
        if (val) {
            item = std::move(*val);
            return true;
        }
        return false;
    }

    /**
     * Checks if queue is empty.
     */
    bool empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    /**
     * Checks if queue is full.
     */
    bool full() const {
        const size_t next_tail = (tail_.load(std::memory_order_acquire) + 1) % Capacity;
        return next_tail == head_.load(std::memory_order_acquire);
    }

    size_t size() const {
        size_t head = head_.load(std::memory_order_acquire);
        size_t tail = tail_.load(std::memory_order_acquire);
        if (tail >= head) return tail - head;
        return Capacity + tail - head;
    }

private:
    // Cache line padding (64 bytes) to prevent false sharing between head and tail
    // This is critical for performance on multi-core systems like Jetson Orin
    static constexpr size_t CACHE_LINE_SIZE = 64;

    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;

    // Buffer storage
    std::array<T, Capacity> buffer_;
};

} // namespace core

