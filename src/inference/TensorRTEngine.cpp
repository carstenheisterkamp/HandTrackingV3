/**
 * V3 TensorRT Engine Wrapper Implementation
 *
 * Supports:
 * - Loading pre-built .engine files
 * - Building engines from .onnx (with caching)
 * - FP16 inference on Jetson
 */

#include "inference/TensorRTEngine.hpp"
#include "core/Logger.hpp"

#include <fstream>
#include <filesystem>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

namespace inference {

// TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            core::Logger::warn("[TensorRT] ", msg);
        }
    }
};

static TRTLogger gLogger;

TensorRTEngine::TensorRTEngine() = default;

TensorRTEngine::~TensorRTEngine() {
    freeBuffers();

    if (context_) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_) {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
}

bool TensorRTEngine::load(const Config& config) {
    modelPath_ = config.modelPath;

    // Determine if we have .engine or .onnx
    std::filesystem::path path(config.modelPath);
    std::string ext = path.extension().string();

    std::string enginePath;

    if (ext == ".engine" || ext == ".trt") {
        enginePath = config.modelPath;
    } else if (ext == ".onnx") {
        // Build engine path (same name, .engine extension)
        enginePath = path.replace_extension(".engine").string();

        // Check if cached engine exists and is newer than ONNX
        bool needsBuild = true;
        if (std::filesystem::exists(enginePath)) {
            auto onnxTime = std::filesystem::last_write_time(config.modelPath);
            auto engineTime = std::filesystem::last_write_time(enginePath);
            needsBuild = (onnxTime > engineTime);
        }

        if (needsBuild) {
            core::Logger::info("Building TensorRT engine from ONNX: ", config.modelPath);
            if (!buildEngine(config.modelPath, enginePath)) {
                core::Logger::error("Failed to build engine from ONNX");
                return false;
            }
        } else {
            core::Logger::info("Using cached TensorRT engine: ", enginePath);
        }
    } else {
        core::Logger::error("Unknown model format: ", ext);
        return false;
    }

    // Load the engine
    if (!loadEngine(enginePath)) {
        return false;
    }

    // Extract tensor info
    extractTensorInfo();

    // Allocate CUDA buffers
    allocateBuffers();

    loaded_ = true;
    core::Logger::info("TensorRT engine loaded successfully");
    core::Logger::info("  Input: ", inputInfo_.name, " [", inputInfo_.dims[0], "x",
                       inputInfo_.dims[1], "x", inputInfo_.dims[2], "x", inputInfo_.dims[3], "]");
    core::Logger::info("  Output: ", outputInfo_.name, " size=", outputInfo_.size);

    return true;
}

bool TensorRTEngine::loadEngine(const std::string& enginePath) {
    // Read engine file
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        core::Logger::error("Cannot open engine file: ", enginePath);
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // Create runtime
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) {
        core::Logger::error("Failed to create TensorRT runtime");
        return false;
    }

    // Deserialize engine
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
    if (!engine_) {
        core::Logger::error("Failed to deserialize engine");
        return false;
    }

    // Create execution context
    context_ = engine_->createExecutionContext();
    if (!context_) {
        core::Logger::error("Failed to create execution context");
        return false;
    }

    return true;
}

bool TensorRTEngine::buildEngine(const std::string& onnxPath, const std::string& enginePath) {
    // Create builder
    auto builder = nvinfer1::createInferBuilder(gLogger);
    if (!builder) {
        core::Logger::error("Failed to create builder");
        return false;
    }

    // Create network with explicit batch (TensorRT 8.5+ default)
    // Note: kEXPLICIT_BATCH is deprecated in 8.5+, explicit batch is now default
    auto network = builder->createNetworkV2(0);
    if (!network) {
        core::Logger::error("Failed to create network");
        delete builder;
        return false;
    }

    // Create ONNX parser
    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser) {
        core::Logger::error("Failed to create ONNX parser");
        delete network;
        delete builder;
        return false;
    }

    // Parse ONNX model
    if (!parser->parseFromFile(onnxPath.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        core::Logger::error("Failed to parse ONNX file");
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Create builder config
    auto config = builder->createBuilderConfig();
    if (!config) {
        core::Logger::error("Failed to create builder config");
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Enable FP16 on Jetson
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        core::Logger::info("FP16 enabled for TensorRT");
    }

    // Set memory pool limit (256 MB)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256 * 1024 * 1024);

    // Build serialized network
    auto serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        core::Logger::error("Failed to build serialized network");
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Save to file
    std::ofstream engineFile(enginePath, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    core::Logger::info("Engine saved to: ", enginePath);

    // Cleanup
    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true;
}

void TensorRTEngine::extractTensorInfo() {
    // TensorRT 8.5+ API uses getNbIOTensors instead of getNbBindings
    int numTensors = engine_->getNbIOTensors();

    for (int i = 0; i < numTensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        auto mode = engine_->getTensorIOMode(name);
        bool isInput = (mode == nvinfer1::TensorIOMode::kINPUT);

        TensorInfo info;
        info.name = name;
        info.isInput = isInput;
        info.size = 1;

        for (int d = 0; d < dims.nbDims; ++d) {
            info.dims.push_back(dims.d[d]);
            info.size *= dims.d[d];
        }

        if (isInput) {
            inputInfo_ = info;
        } else {
            outputInfo_ = info;
        }
    }
}

void TensorRTEngine::allocateBuffers() {
    size_t inputBytes = inputInfo_.size * sizeof(float);
    size_t outputBytes = outputInfo_.size * sizeof(float);

    cudaMalloc(&d_input_, inputBytes);
    cudaMalloc(&d_output_, outputBytes);

    core::Logger::info("CUDA buffers allocated: input=", inputBytes, " output=", outputBytes);
}

void TensorRTEngine::freeBuffers() {
    if (d_input_) {
        cudaFree(d_input_);
        d_input_ = nullptr;
    }
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
}

bool TensorRTEngine::infer(const void* inputData, void* outputData) {
    if (!loaded_) {
        core::Logger::error("Engine not loaded");
        return false;
    }

    // Copy input to GPU
    size_t inputBytes = inputInfo_.size * sizeof(float);
    cudaMemcpy(d_input_, inputData, inputBytes, cudaMemcpyHostToDevice);

    // Run inference
    void* bindings[] = {d_input_, d_output_};
    bool success = context_->executeV2(bindings);

    if (!success) {
        core::Logger::error("Inference failed");
        return false;
    }

    // Copy output to host
    size_t outputBytes = outputInfo_.size * sizeof(float);
    cudaMemcpy(outputData, d_output_, outputBytes, cudaMemcpyDeviceToHost);

    return true;
}

bool TensorRTEngine::inferAsync(void** bindings, void* stream) {
    if (!loaded_) {
        return false;
    }

    cudaStream_t cudaStream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    // TensorRT 8.5+: Use setTensorAddress + enqueueV3
    // Set input tensor address
    context_->setTensorAddress(inputInfo_.name.c_str(), bindings[0]);
    // Set output tensor address
    context_->setTensorAddress(outputInfo_.name.c_str(), bindings[1]);

    return context_->enqueueV3(cudaStream);
}

} // namespace inference

