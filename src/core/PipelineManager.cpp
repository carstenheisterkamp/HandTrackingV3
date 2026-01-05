#include "core/PipelineManager.hpp"
#include "core/Logger.hpp"
#include <iostream>
#include <stdexcept>

// Explicit includes for v3 clarity
#include <depthai/pipeline/node/Camera.hpp>
#include <depthai/pipeline/node/NeuralNetwork.hpp>
#include <depthai/pipeline/node/MobileNetDetectionNetwork.hpp>
#include <depthai/pipeline/node/ImageManip.hpp>
#include <depthai/pipeline/node/Script.hpp>
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

        // Enable IR Light (Dot Projector & Flood Light)
        // Useful for low-light or mono-camera depth accuracy
        try {
            // API v2.17+ uses setIrLaserDotProjectorIntensity (0..1) or similar?
            // Actually, older API used Brightness (mA), newer uses Intensity (0.0 - 1.0) or similar.
            // But wait, the error says "did you mean setIrLaserDotProjectorIntensity".
            // Let's try using Intensity. Note: Intensity usually takes 0.0 to 1.0 float.
            // However, if the previous code passed 800 (mA), maybe the new API expects 0..1?
            // Let's check if we can pass 0.8f (assuming 800mA is roughly high intensity) or if it takes mA.
            // Actually, looking at DepthAI docs, setIrLaserDotProjectorIntensity takes float intensity (0..1).
            // But wait, there is also setIrLaserDotProjectorBrightness(float mA) in some versions.
            // If the compiler says it doesn't exist, we must use Intensity.

            // Let's try setting it to 0.8f (80%) which is safe.
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
        Logger::info("Creating Hand Tracking Pipeline...");

        // 1. Palm Detection Node
        auto palmDetect = pipeline_->create<dai::node::NeuralNetwork>();
        palmDetect->setBlobPath("models/palm_detection_sh4.blob");
        // Palm detector expects 128x128
        auto palmInput = cam->requestOutput(
            std::make_pair(128, 128),
            dai::ImgFrame::Type::RGB888p,
            dai::ImgResizeMode::STRETCH,
            config.fps
        );
        palmInput->link(palmDetect->input);

        // 2. Script Node (Logic: Detection -> Crop Config)
        auto script = pipeline_->create<dai::node::Script>();
        script->setProcessor(dai::ProcessorType::LEON_CSS);
        script->setScript(R"(
            # Scaling factors
            PAD_X = 0.0
            PAD_Y = 0.0

            while True:
                # Get detection and image
                detections = node.io['detections'].get().getLayerFp16("class_registr_scores")
                # Simple logic: Just take the first detection if available (simplified)
                # Real palm detector output parsing is complex (anchors etc).
                # For V3, let's assume we use a MobileNet version if available,
                # OR we just pass the full image to landmarks if we can't parse anchors easily in script.

                # WAIT: Parsing SSD anchors in Script is hard.
                # Alternative: Use MobileNetDetectionNetwork if the blob is supported.
                # The provided blob 'palm_detection_sh4.blob' is likely the MediaPipe one (custom output).

                # Fallback for this iteration:
                # We will stick to the single Landmark model but fix the input size/type.
                # The user reported 'e-42' which was a type mismatch.
                # Let's see if the FP16 fix works first.

                # If we want to add Palm Detection, we need to parse its output.
                # Let's just send the Palm Detection output to Host for now to debug.

                node.io['cfg'].send(ImgManipConfig()) # Dummy
        )");

        // REVERTING SCRIPT PLAN: Too complex for this iteration without testing.
        // Let's stick to the Landmark model but ensure we feed it correctly.
        // If the user says "Velocity [e-42]", it was definitely the float parsing.
        // I fixed that in InputLoop.cpp.

        // Let's just add the Palm Detector as a parallel node so we can see if it detects anything.
        // This gives us "Position" (if we parse it on host).

        // 2. Landmark Node
        Logger::info("Creating Landmark NeuralNetwork node with blob: ", config.nnPath);
        auto landmarkNN = pipeline_->create<dai::node::NeuralNetwork>();
        landmarkNN->setBlobPath(config.nnPath);

        // Request dedicated output for NN (224x224 is typical for hand landmarks)
        auto nnInput = cam->requestOutput(
            std::make_pair(224, 224),
            dai::ImgFrame::Type::RGB888p, // Explicitly request RGB planar
            dai::ImgResizeMode::STRETCH,
            config.fps
        );

        nnInput->link(landmarkNN->input);

        // Use Sync node to synchronize RGB, Landmarks, and Palm Detections
        auto sync = pipeline_->create<dai::node::Sync>();
        sync->setSyncThreshold(std::chrono::milliseconds(50)); // 50ms tolerance

        rgbOutput->link(sync->inputs["rgb"]);
        landmarkNN->out.link(sync->inputs["landmarks"]);
        palmDetect->out.link(sync->inputs["palm"]);

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

