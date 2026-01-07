#include "core/PipelineManager.hpp"
#include "core/Logger.hpp"
#include <iostream>
#include <stdexcept>

// Explicit includes for v3 clarity
#include <depthai/pipeline/node/Camera.hpp>
#include <depthai/pipeline/node/NeuralNetwork.hpp>
#include <depthai/pipeline/node/StereoDepth.hpp>
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

    bool connected = false;
    int retries = 0;
    const int MAX_RETRIES = 10; // 10 Versuche
    const int RETRY_DELAY_MS = 2000; // 2 Sekunden Pause

    while (!connected && retries < MAX_RETRIES) {
        try {
            // 1. Create Device
            if (retries > 0) Logger::info("Attempting to connect to OAK device (Attempt ", retries + 1, "/", MAX_RETRIES, ")...");

            // PoE device connection: Pass IP directly to Device constructor
            if (!config.deviceIp.empty()) {
                Logger::info("Connecting to PoE device at: ", config.deviceIp);
                device_ = std::make_shared<dai::Device>(config.deviceIp);
            } else {
                // Fallback: Try any available device
                device_ = std::make_shared<dai::Device>();
            }
            connected = true;

        } catch (const std::exception& e) {
            retries++;
            Logger::warn("Failed to connect to device: ", e.what());
            if (retries < MAX_RETRIES) {
                Logger::warn("Retrying in ", RETRY_DELAY_MS, "ms...");
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
            }
        }
    }

    if (!connected) {
        Logger::error("Fatal error: Could not connect to OAK device after ", MAX_RETRIES, " attempts.");
        throw std::runtime_error("Device connection failed after retries.");
    }

    try {
        // 2. Create Pipeline with device
        pipeline_ = std::make_unique<dai::Pipeline>(device_);

        // NOTE: SIPP buffer settings removed - no longer needed without StereoDepth on device

        // 3. Create Nodes and Queues
        createPipeline(config);

        Logger::info("DepthAI Device initialized: ", device_->getDeviceId());

        // Handle PoE devices which return UNKNOWN for USB speed
        if (device_->getUsbSpeed() == dai::UsbSpeed::UNKNOWN) {
            Logger::info("Connection: Ethernet (PoE) / Unknown");
        } else {
            Logger::info("Connection: USB ", device_->getUsbSpeed());
        }

        // Enable IR Light (Dot Projector & Flood Light)
        // Useful for low-light or mono-camera depth accuracy
        try {
            // Let's try setting it to 0.8f (80%) which is safe.
            // Updated for v3 API (Intensity 0..1 instead of Brightness mA)
            device_->setIrLaserDotProjectorIntensity(0.8f);
            device_->setIrFloodLightIntensity(0.8f);
            Logger::info("IR Light enabled (Intensity: 0.8)");
        } catch (const std::exception& e) {
            Logger::warn("Failed to enable IR light (Device might not support it): ", e.what());
        }

        // Log connected cameras
        auto cameras = device_->getConnectedCameras();
        for(const auto& cam : cameras) {
            std::string name;
            switch(cam) {
                case dai::CameraBoardSocket::CAM_A: name = "RGB (Center)"; break;
                case dai::CameraBoardSocket::CAM_B: name = "Mono (Left)"; break;
                case dai::CameraBoardSocket::CAM_C: name = "Mono (Right)"; break;
                default: name = "Other"; break;
            }
            Logger::info("Connected camera: ", cam, " [", name, "]");
        }

    } catch (const std::exception& e) {
        Logger::error("Failed to initialize device: ", e.what());
        throw;
    }
}

void PipelineManager::createPipeline(const Config& config) {
    Logger::debug("Creating pipeline nodes...");

    // ============================================================
    // STABLE PIPELINE: RGB + Palm Detection + Hand Landmarks
    // ============================================================

    // RGB Camera
    auto cam = pipeline_->create<dai::node::Camera>()->build(
        dai::CameraBoardSocket::CAM_A,
        std::make_pair(1920, 1080),
        config.fps
    );

    // Preview output for visualization
    auto rgbOutput = cam->requestOutput(
        std::make_pair(config.previewWidth, config.previewHeight),
        dai::ImgFrame::Type::NV12,
        dai::ImgResizeMode::CROP,
        config.fps
    );

    if (!config.nnPath.empty()) {
        Logger::info("Creating Hand Tracking Pipeline (Palm + Landmarks)...");

        // 1. Palm Detection Node (using sh4 blob)
        auto palmDetect = pipeline_->create<dai::node::NeuralNetwork>();
        palmDetect->setBlobPath("models/palm_detection_sh4.blob");
        // Warning: "Number of inference threads assigned for network is 1, assigning 2 will likely yield in better performance"
        palmDetect->setNumInferenceThreads(2);
        palmDetect->setNumNCEPerInferenceThread(1); // Explicitly limit NCE to 1 per thread to avoid resource exhaustion

        auto palmInput = cam->requestOutput(
            std::make_pair(128, 128),
            dai::ImgFrame::Type::BGR888p,  // PLANAR format (CHW) for NN compatibility
            dai::ImgResizeMode::STRETCH,
            config.fps
        );
        palmInput->link(palmDetect->input);

        // 2. Landmark Node (using sh4 blob)
        auto landmarkNN = pipeline_->create<dai::node::NeuralNetwork>();
        landmarkNN->setBlobPath(config.nnPath);
        landmarkNN->setNumInferenceThreads(2); // Attempting 2 threads for landmarks as well for throughput
        landmarkNN->setNumNCEPerInferenceThread(1); // Explicitly limit NCE to 1 per thread

        auto nnInput = cam->requestOutput(
            std::make_pair(224, 224),
            dai::ImgFrame::Type::BGR888p,  // PLANAR format (CHW) for NN compatibility
            dai::ImgResizeMode::STRETCH,
            config.fps
        );
        nnInput->link(landmarkNN->input);

        // Sync node: RGB + Palm + Landmarks
        auto sync = pipeline_->create<dai::node::Sync>();
        sync->setSyncThreshold(std::chrono::milliseconds(20));

        rgbOutput->link(sync->inputs["rgb"]);
        palmDetect->out.link(sync->inputs["palm"]);
        landmarkNN->out.link(sync->inputs["landmarks"]);

        auto syncQueue = sync->out.createOutputQueue();
        queues_["sync"] = syncQueue;

        Logger::info("Pipeline: RGB + Palm + Landmarks");
    } else {
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
         try {
            // Clear queues first to stop data flow
            queues_.clear();

            // Close device connection
            device_->close();

            // Reset shared pointers
            device_.reset();
            pipeline_.reset();

            // Give the device time to fully disconnect
            // This prevents "device busy" on quick restart
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            Logger::info("Device closed successfully.");
        } catch (const std::exception& e) {
            Logger::error("Error during device shutdown: ", e.what());
            // Force reset even on error
            device_.reset();
            pipeline_.reset();
        }
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

