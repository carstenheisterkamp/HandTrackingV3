#include "core/PipelineManager.hpp"
#include "core/Logger.hpp"
#include <iostream>
#include <stdexcept>

// Explicit includes for v3 clarity
#include <depthai/pipeline/node/Camera.hpp>
#include <depthai/pipeline/node/NeuralNetwork.hpp>
#include <depthai/pipeline/node/Sync.hpp>
#include <depthai/pipeline/node/ImageManip.hpp>

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
    const int RETRY_DELAY_MS = 3000; // 3 Sekunden Pause

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


        // 3. Create Nodes and Queues
        createPipeline(config);

        Logger::info("DepthAI Device initialized: ", device_->getDeviceId());

        // Suppress benign warnings about NN input format mismatch
        device_->setLogLevel(dai::LogLevel::ERR);
        device_->setLogOutputLevel(dai::LogLevel::ERR);

        // Handle PoE devices which return UNKNOWN for USB speed
        if (device_->getUsbSpeed() == dai::UsbSpeed::UNKNOWN) {
            Logger::info("Connection: Ethernet (PoE) / Unknown");
        } else {
            Logger::info("Connection: USB ", device_->getUsbSpeed());
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

    // AUTO MODE - Let camera handle focus and exposure automatically
    // Manual settings caused overexposure and FPS drops
    cam->initialControl.setAutoFocusMode(dai::CameraControl::AutoFocusMode::CONTINUOUS_VIDEO);
    cam->initialControl.setAutoExposureEnable();
    cam->initialControl.setAutoWhiteBalanceMode(dai::CameraControl::AutoWhiteBalanceMode::AUTO);

    Logger::info("Camera config: AUTO mode (Focus/Exposure/WB)");

    // Preview output for visualization
    auto rgbOutput = cam->requestOutput(
        std::make_pair(config.previewWidth, config.previewHeight),
        dai::ImgFrame::Type::NV12,
        dai::ImgResizeMode::CROP,
        config.fps
    );

    if (!config.nnPath.empty()) {
        Logger::info("Creating Hand Tracking Pipeline (Palm + Landmarks)...");

        // 1. Palm Detection Node (using FULL sh4 blob)
        auto palmDetect = pipeline_->create<dai::node::NeuralNetwork>();
        palmDetect->setBlobPath("models/palm_detection_full_sh4.blob");
        palmDetect->setNumInferenceThreads(1);  // REDUCED: 1 thread is faster on Myriad X
        palmDetect->setNumNCEPerInferenceThread(1);

        // Explicit ImageManip for Palm Detection (192x192 RGB Planar)
        auto manipPalm = pipeline_->create<dai::node::ImageManip>();
        manipPalm->initialConfig->setOutputSize(192, 192);
        manipPalm->initialConfig->base.resizeMode = dai::ImageManipConfig::ResizeMode::LETTERBOX;
        manipPalm->initialConfig->setFrameType(dai::ImgFrame::Type::BGR888p);

        // Link Camera Preview (RGB Output) -> Manip -> NN
        rgbOutput->link(manipPalm->inputImage);
        manipPalm->out.link(palmDetect->input);

        // 2. Landmark Node (using sh4 blob) - DIRECT FEED, NO SCRIPT
        auto landmarkNN = pipeline_->create<dai::node::NeuralNetwork>();
        landmarkNN->setBlobPath("models/hand_landmark_full_sh4.blob");
        landmarkNN->setNumInferenceThreads(1);  // REDUCED: 1 thread is faster
        landmarkNN->setNumNCEPerInferenceThread(1);

        // Direct ImageManip for Landmark (224x224 from full RGB)
        auto manipLandmark = pipeline_->create<dai::node::ImageManip>();
        manipLandmark->initialConfig->setOutputSize(224, 224);
        manipLandmark->initialConfig->base.resizeMode = dai::ImageManipConfig::ResizeMode::LETTERBOX;
        manipLandmark->initialConfig->setFrameType(dai::ImgFrame::Type::BGR888p);

        // Direct wiring: Camera -> Manip -> Landmark NN
        rgbOutput->link(manipLandmark->inputImage);
        manipLandmark->out.link(landmarkNN->input);

        // Sync node: RGB + Palm + Landmarks
        auto sync = pipeline_->create<dai::node::Sync>();
        sync->setSyncThreshold(std::chrono::milliseconds(10));  // REDUCED from 20ms

        rgbOutput->link(sync->inputs["rgb"]);
        palmDetect->out.link(sync->inputs["palm"]);
        landmarkNN->out.link(sync->inputs["landmarks"]);

        // Create queue immediately (v3 API allows this before start())
        auto syncQueue = sync->out.createOutputQueue(4, false);
        queues_["sync"] = syncQueue;

        Logger::info("Pipeline: RGB + Palm + Landmarks + Stereo (Mono L/R)");
    } else {
        // Fallback: RGB only (no NN)
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
            // 1. Clear queues first to stop data flow
            queues_.clear();
            Logger::debug("Queues cleared.");

            // 2. Request device reset before closing (helps with PoE reconnect)
            if (!device_->isClosed()) {
                try {
                    // Send reset command to device firmware
                    // This ensures clean state for next connection
                    Logger::debug("Requesting device reset...");
                    // Note: In DepthAI v3, close() already handles graceful shutdown
                    // Explicit reset via bootloader would be: device->flashBootloader(...)
                    // but that's too aggressive for normal operation
                } catch (const std::exception& resetErr) {
                    Logger::warn("Device reset command failed (not critical): ", resetErr.what());
                }

                // 3. Close device connection
                device_->close();
                Logger::debug("Device closed.");
            }

            // 4. Reset shared pointers (order matters: pipeline before device)
            pipeline_.reset();
            device_.reset();
            Logger::debug("Pointers reset.");

            // 5. Give OAK-D time to fully reset firmware and release PoE resources
            // PoE devices need more time than USB devices for network stack cleanup
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));

            Logger::info("Device closed successfully. Wait 2s before reconnect.");
        } catch (const std::exception& e) {
            Logger::error("Error during device shutdown: ", e.what());
            // Force reset even on error
            queues_.clear();
            pipeline_.reset();
            device_.reset();
            // Still wait to give device time to recover
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        }
    }
}

void PipelineManager::update(const std::string& nodeName, const std::string& param, float value) {
    // DepthAI v3 Runtime Configuration
    Logger::info("Update requested: ", nodeName, ".", param, " = ", value, " (Not implemented yet)");
}

std::shared_ptr<dai::MessageQueue> PipelineManager::getOutputQueue(const std::string& name, int maxSize, bool blocking) {
    // Queues are now created during pipeline construction
    auto it = queues_.find(name);
    if (it != queues_.end()) {
        return it->second;
    }

    Logger::warn("Queue '", name, "' not found. Available queues: ");
    for (const auto& [qname, q] : queues_) {
        Logger::warn("  - ", qname);
    }
    return nullptr;
}

} // namespace core

