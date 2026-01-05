#include "core/PipelineManager.hpp"
#include "core/Logger.hpp"
#include <iostream>
#include <stdexcept>

// Explicit includes for v3 clarity
#include <depthai/pipeline/node/Camera.hpp>
#include <depthai/pipeline/node/NeuralNetwork.hpp>
#include <depthai/pipeline/node/Sync.hpp>

namespace core {

PipelineManager::PipelineManager() {
    // Constructor
}

PipelineManager::~PipelineManager() {
    stop();
}

void PipelineManager::init(const Config& config) {
    Logger::info("Initializing PipelineManager...");
    currentConfig_ = config;

    try {
        // 1. Create Device first
        device_ = std::make_shared<dai::Device>();

        // 2. Create Pipeline with device
        pipeline_ = std::make_unique<dai::Pipeline>(device_);

        // 3. Create Nodes and Queues
        createPipeline(config);

        Logger::info("DepthAI Device initialized: ", device_->getDeviceId());
        Logger::info("USB Speed: ", device_->getUsbSpeed());

        // Log connected cameras
        auto cameras = device_->getConnectedCameras();
        for(const auto& cam : cameras) {
            Logger::info("Connected camera: ", cam);
        }

    } catch (const std::exception& e) {
        Logger::error("Failed to initialize device: ", e.what());
        throw;
    }
}

void PipelineManager::createPipeline(const Config& config) {
    Logger::debug("Creating pipeline nodes...");

    // Use 'Camera' node
    // In v3, we configure via build() and requestOutput()
    // build() takes boardSocket, resolution, fps
    auto cam = pipeline_->create<dai::node::Camera>()->build(
        dai::CameraBoardSocket::CAM_A,
        std::make_pair(1920, 1080), // Sensor resolution
        config.fps
    );

    // Output
    // Request output for the preview stream
    // requestOutput takes size (width, height), type, resizeMode, fps
    // We use this to handle ISP scaling/resizing
    auto rgbOutput = cam->requestOutput(
        std::make_pair(config.previewWidth, config.previewHeight),
        std::nullopt, // type (auto)
        dai::ImgResizeMode::CROP, // or STRETCH/LETTERBOX
        config.fps
    );

    if (!config.nnPath.empty()) {
        Logger::info("Creating NeuralNetwork node with blob: ", config.nnPath);
        auto nn = pipeline_->create<dai::node::NeuralNetwork>();
        nn->setBlobPath(config.nnPath);

        // Request dedicated output for NN (224x224 is typical for hand landmarks)
        // TODO: Make NN input size configurable
        auto nnInput = cam->requestOutput(
            std::make_pair(224, 224),
            std::nullopt,
            dai::ImgResizeMode::STRETCH,
            config.fps
        );

        nnInput->link(nn->input);

        // Use Sync node to synchronize RGB and NN results
        auto sync = pipeline_->create<dai::node::Sync>();
        sync->setSyncThreshold(std::chrono::milliseconds(50)); // 50ms tolerance

        rgbOutput->link(sync->inputs["rgb"]);
        nn->out.link(sync->inputs["nn"]);

        auto syncQueue = sync->out.createOutputQueue();
        queues_["sync"] = syncQueue;
    } else {
        // Create the queue immediately if no NN
        auto rgbQueue = rgbOutput->createOutputQueue();
        queues_["rgb"] = rgbQueue;
    }

    Logger::debug("Pipeline creation complete.");
}

void PipelineManager::start() {
    if (!pipeline_) {
        throw std::runtime_error("Pipeline not initialized. Call init() first.");
    }
    Logger::info("Starting pipeline...");
    pipeline_->start();
    Logger::info("Pipeline started.");
}

void PipelineManager::stop() {
    if (device_) {
        Logger::info("Stopping pipeline and closing device...");
        // pipeline_->stop(); // If available, otherwise device close handles it
        device_->close();
        device_.reset();
        pipeline_.reset();
        queues_.clear();
        Logger::info("Device closed.");
    }
}

void PipelineManager::update(const std::string& nodeName, const std::string& param, float value) {
    // DepthAI v3 Runtime Configuration
    Logger::info("Update requested: ", nodeName, ".", param, " = ", value, " (Not implemented yet)");
}

std::shared_ptr<dai::MessageQueue> PipelineManager::getOutputQueue(const std::string& name, int maxSize, bool blocking) {
    if (queues_.find(name) != queues_.end()) {
        return queues_[name];
    }

    Logger::warn("Queue '", name, "' not found in managed queues.");
    return nullptr;
}

} // namespace core

