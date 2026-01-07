#include "core/StereoKernel.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace core {

__global__ void stereoSADKernel(const uint8_t* __restrict__ left,
                                const uint8_t* __restrict__ right,
                                uint16_t* __restrict__ depth,
                                int width, int height, int maxDisp) {
    // 2D Thread Index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Constants for OAK-D (Approximate)
    // Baseline: 75mm (7.5cm)
    // FOV: ~72 deg -> Focal Length ~ 440px for 640 width
    // BF = Baseline * Focal = 75 * 440 = 33000
    const float bf = 33000.0f;

    // Block Matching Parameters
    const int RADIUS = 3; // 7x7 Window
    // Bounds check for window
    if (x < maxDisp + RADIUS || x >= width - RADIUS || y < RADIUS || y >= height - RADIUS) {
        depth[y * width + x] = 0;
        return;
    }

    int bestDisp = 0;
    unsigned int minSAD = 999999; // Max Int

    // Search Disparity Range
    // Hint: Loop could be unrolled or optimized with shared mem,
    // but global mem is fast enough for 640x480 on Orin.
    for (int d = 0; d < maxDisp; ++d) {
        unsigned int sad = 0;

        // Sum of Absolute Differences (SAD) for 7x7 window
        for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
            for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                // Pixel coords
                int lx = x + dx;
                int ly = y + dy;
                int rx = lx - d; // Right image pixel is shifted by 'd'

                // Load pixels
                int lVal = left[ly * width + lx];
                int rVal = right[ly * width + rx];

                // __sad intrinsic optimized for integers
                sad += abs(lVal - rVal);
            }
        }

        // Winner Takes All
        if (sad < minSAD) {
            minSAD = sad;
            bestDisp = d;
        }
    }

    // Sub-pixel refinement (optional, skipped for perf/simplicity)

    // Convert to Depth (mm)
    // Z = BF / Disparity
    if (bestDisp > 3 && minSAD < 2000) { // Thresholds to remove noise
        depth[y * width + x] = (uint16_t)(bf / (float)bestDisp);
    } else {
        depth[y * width + x] = 0;
    }
}

void computeStereoDepth(const uint8_t* d_left, const uint8_t* d_right,
                        uint16_t* d_depth,
                        int width, int height,
                        int disparityRange) {

    // Block size 32x16 is standard efficiency
    dim3 blockSize(32, 16);
    // Grid covers the whole image
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch Kernel
    stereoSADKernel<<<gridSize, blockSize>>>(d_left, d_right, d_depth, width, height, disparityRange);

    // No sync here, assume stream 0 or caller syncs.
    // Errors will be caught by subsequent CUDA calls.
}

}

