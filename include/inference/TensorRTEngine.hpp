#pragma once

#include <string>
#include <vector>
#include <memory>

// Forward declarations for TensorRT
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace inference {

/**
 * V3 TensorRT Engine Wrapper
 *
 * Handles:
 * - Loading .engine files (or building from .onnx)
 * - CUDA memory management for input/output
 * - Synchronous inference execution
 */
class TensorRTEngine {
public:
    struct Config {
        std::string modelPath;      // Path to .onnx or .engine file
        int maxBatchSize = 1;       // Always 1 for real-time
        bool fp16 = true;           // Use FP16 on Jetson
        int dlaCore = -1;           // -1 = GPU, 0/1 = DLA core
    };

    struct TensorInfo {
        std::string name;
        std::vector<int> dims;
        size_t size;                // Total elements
        bool isInput;
    };

    TensorRTEngine();
    ~TensorRTEngine();

    // Non-copyable
    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    /**
     * Load engine from file or build from ONNX
     * @return true on success
     */
    bool load(const Config& config);

    /**
     * Run inference
     * @param inputData Pointer to input data (must match input tensor size)
     * @param outputData Pointer to output buffer (must match output tensor size)
     * @return true on success
     */
    bool infer(const void* inputData, void* outputData);

    /**
     * Run inference with multiple inputs/outputs
     * @param bindings Array of device pointers for all tensors
     * @return true on success
     */
    bool inferAsync(void** bindings, void* stream = nullptr);

    /**
     * Get input tensor info
     */
    [[nodiscard]] const TensorInfo& getInputInfo() const { return inputInfo_; }

    /**
     * Get output tensor info
     */
    [[nodiscard]] const TensorInfo& getOutputInfo() const { return outputInfo_; }

    /**
     * Check if engine is loaded
     */
    [[nodiscard]] bool isLoaded() const { return loaded_; }

    /**
     * Get model path
     */
    [[nodiscard]] const std::string& getModelPath() const { return modelPath_; }

private:
    std::string modelPath_;
    bool loaded_ = false;

    // TensorRT objects (using raw pointers for PIMPL pattern simplicity)
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // Tensor info
    TensorInfo inputInfo_;
    TensorInfo outputInfo_;

    // CUDA buffers
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;

    // Helper methods
    bool loadEngine(const std::string& enginePath);
    bool buildEngine(const std::string& onnxPath, const std::string& enginePath);
    void extractTensorInfo();
    void allocateBuffers();
    void freeBuffers();
};

} // namespace inference

