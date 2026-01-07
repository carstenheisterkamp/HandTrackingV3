#pragma once
#include <cstdint>

namespace core {

/**
 * Computes depth map from stereo pair using Block Matching (SAD) in CUDA.
 *
 * @param d_left Device pointer to Left image (Gray8)
 * @param d_right Device pointer to Right image (Gray8)
 * @param d_depth Device pointer to Output Depth map (Uint16, mm)
 * @param width Image width
 * @param height Image height
 * @param disparityRange Max disparity to search (default 64)
 */
void computeStereoDepth(const uint8_t* d_left, const uint8_t* d_right,
                        uint16_t* d_depth,
                        int width, int height,
                        int disparityRange = 64);

}

