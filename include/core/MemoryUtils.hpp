#pragma once

#include <memory>
#include <cstdlib>
#include <stdexcept>
#include "core/Logger.hpp"

// Forward declaration for CUDA runtime API if not available
// We will use a macro to enable CUDA specific code
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace core {

struct AlignedDeleter {
    void operator()(void* ptr) const {
#ifdef ENABLE_CUDA
        // If the buffer was registered, we should unregister it.
        // However, tracking which buffer is registered is complex here.
        // Ideally, we unregister before freeing.
        // For now, we assume the user handles registration/unregistration
        // or we just free the memory.
        // cudaHostUnregister(ptr);
#endif
        std::free(ptr);
    }
};

/**
 * Allocates memory aligned to the specified boundary.
 * Default alignment is 256 bytes (optimal for GPU DMA).
 */
template<typename T>
std::unique_ptr<T, AlignedDeleter> allocate_aligned(size_t count, size_t alignment = 256) {
    size_t size = count * sizeof(T);
    // Ensure size is a multiple of alignment for safety
    if (size % alignment != 0) {
        size = ((size / alignment) + 1) * alignment;
    }

    void* ptr = std::aligned_alloc(alignment, size);
    if (!ptr) {
        throw std::bad_alloc();
    }

    return std::unique_ptr<T, AlignedDeleter>(static_cast<T*>(ptr));
}

/**
 * Registers a buffer for Zero-Copy access with CUDA.
 * This allows the GPU to access the host memory directly (Unified Memory).
 */
inline void register_buffer_cuda(void* ptr, size_t size) {
#ifdef ENABLE_CUDA
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterMapped);
    if (err == cudaSuccess) {
        Logger::debug("Registered buffer for CUDA Zero-Copy: ", ptr, " Size: ", size);
    } else if (err == cudaErrorHostMemoryAlreadyRegistered) {
        // Buffer already registered (e.g., from previous run), this is OK
        Logger::debug("Buffer already registered for CUDA: ", ptr);
    } else {
        Logger::error("cudaHostRegister failed: ", cudaGetErrorString(err));
        // Don't throw - continue without zero-copy for this buffer
        // throw std::runtime_error("Failed to register CUDA buffer");
    }
#else
    (void)ptr;
    (void)size;
    Logger::debug("CUDA not enabled. Skipping buffer registration.");
#endif
}

/**
 * Gets the device pointer for a registered host pointer.
 */
inline void* get_device_pointer(void* hostPtr) {
#ifdef ENABLE_CUDA
    void* devPtr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&devPtr, hostPtr, 0);
    if (err != cudaSuccess) {
        Logger::error("cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
        return nullptr;
    }
    return devPtr;
#else
    (void)hostPtr;
    return nullptr;
#endif
}

} // namespace core

