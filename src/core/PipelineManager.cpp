#include "core/PipelineManager.hpp"
#include "core/Logger.hpp"
#include <stdexcept>

// V3: Sensor-only pipeline - only Camera and Sync nodes
#include <depthai/pipeline/node/Camera.hpp>
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
    const int MAX_RETRIES = 10; // 10 Attempts
    const int RETRY_DELAY_MS = 5000; // 5 seconds (increased for PoE stability)

    while (!connected && retries < MAX_RETRIES) {
        try {
            if (retries > 0) {
                Logger::info("Attempting to connect to OAK device (Attempt ", retries + 1, "/", MAX_RETRIES, ")...");

                // On PoE devices, first retry might fail due to stale ARP cache
                // Longer delay helps network stack reset
                if (retries == 1 && !config.deviceIp.empty()) {
                    Logger::warn("First retry - if this fails repeatedly, consider:");
                    Logger::warn("  1. sudo arp -d ", config.deviceIp, "  (clear ARP cache)");
                    Logger::warn("  2. Power cycle the OAK-D (unplug PoE for 10s)");
                    Logger::warn("  3. Check network: ping ", config.deviceIp);
                }
            }

            // PoE device connection: Pass IP directly to Device constructor
            if (!config.deviceIp.empty()) {
                Logger::info("Connecting to PoE device at: ", config.deviceIp);
                device_ = std::make_shared<dai::Device>(config.deviceIp);
            } else {
                // Fallback: Try any available device
                Logger::info("Connecting to any available device (USB/PoE auto-detect)...");
                device_ = std::make_shared<dai::Device>();
            }
            connected = true;
            Logger::info("Successfully connected to device!");

        } catch (const std::exception& e) {
            retries++;
            Logger::warn("Connection failed: ", e.what());
            if (retries < MAX_RETRIES) {
                Logger::warn("Waiting ", RETRY_DELAY_MS, "ms before retry ", retries + 1, "/", MAX_RETRIES, "...");
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
            } else {
                Logger::error("Max retries reached. Connection failed.");
            }
        }
    }

    if (!connected) {
        Logger::error("═══════════════════════════════════════════════");
        Logger::error("FATAL: Could not connect to OAK device!");
        Logger::error("═══════════════════════════════════════════════");
        Logger::error("Troubleshooting steps:");
        Logger::error("  1. Check power: Is OAK-D LED on?");
        Logger::error("  2. Check network: ping ", config.deviceIp.empty() ? "169.254.1.222" : config.deviceIp);
        Logger::error("  3. Clear ARP: sudo arp -d ", config.deviceIp.empty() ? "169.254.1.222" : config.deviceIp);
        Logger::error("  4. Power cycle: Unplug PoE for 10 seconds");
        Logger::error("  5. Last resort: Reboot Jetson");
        Logger::error("═══════════════════════════════════════════════");
        throw std::runtime_error("Device connection failed after " + std::to_string(MAX_RETRIES) + " retries.");
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
    // V3 SENSOR-ONLY PIPELINE: RGB + optional Mono L/R
    // Per OPTIMAL_WORKFLOW_V3.md:
    // - OAK-D is a pure sensor (no NNs, no encoding, no tracker)
    // - All NNs run on Jetson (TensorRT)
    // - Mono L/R preserved for stereo depth on Jetson (optional)
    // ============================================================

    // ─────────────────────────────────────────────────────────────
    // 1. RGB Camera (Center) - Preview for NN input
    // ─────────────────────────────────────────────────────────────
    auto camRgb = pipeline_->create<dai::node::Camera>()->build(
        dai::CameraBoardSocket::CAM_A,
        std::make_pair(1920, 1080),
        config.fps
    );

    // AUTO MODE with Exposure Limit (guarantees FPS target)
    camRgb->initialControl.setAutoFocusMode(dai::CameraControl::AutoFocusMode::CONTINUOUS_VIDEO);
    camRgb->initialControl.setAutoExposureEnable();
    camRgb->initialControl.setAutoExposureLimit(20000); // 20ms exposure limit for 30+ FPS
    camRgb->initialControl.setAutoExposureCompensation(2);
    camRgb->initialControl.setAutoWhiteBalanceMode(dai::CameraControl::AutoWhiteBalanceMode::AUTO);

    Logger::info("RGB Camera: 1080p @ ", config.fps, " FPS (Exposure Limit: 20ms)");

    // RGB Preview: 640×360 NV12 (Zero-Copy friendly, LETTERBOX for NN)
    auto rgbPreview = camRgb->requestOutput(
        std::make_pair(config.previewWidth, config.previewHeight),
        dai::ImgFrame::Type::NV12,
        dai::ImgResizeMode::LETTERBOX,
        config.fps
    );

    Logger::info("RGB Preview: ", config.previewWidth, "x", config.previewHeight, " NV12 LETTERBOX");

    if (config.enableStereo) {
        // ─────────────────────────────────────────────────────────────
        // 2. Mono Left Camera (for Stereo Depth) - OPTIONAL
        // ─────────────────────────────────────────────────────────────
        auto camMonoLeft = pipeline_->create<dai::node::Camera>()->build(
            dai::CameraBoardSocket::CAM_B,
            std::make_pair(640, 400),
            config.fps
        );
        camMonoLeft->initialControl.setAutoExposureEnable();
        camMonoLeft->initialControl.setAutoExposureLimit(20000);

        auto monoLeftOutput = camMonoLeft->requestOutput(
            std::make_pair(640, 400),
            dai::ImgFrame::Type::GRAY8,
            dai::ImgResizeMode::CROP,
            config.fps
        );

        // ─────────────────────────────────────────────────────────────
        // 3. Mono Right Camera (for Stereo Depth) - OPTIONAL
        // ─────────────────────────────────────────────────────────────
        auto camMonoRight = pipeline_->create<dai::node::Camera>()->build(
            dai::CameraBoardSocket::CAM_C,
            std::make_pair(640, 400),
            config.fps
        );
        camMonoRight->initialControl.setAutoExposureEnable();
        camMonoRight->initialControl.setAutoExposureLimit(20000);

        auto monoRightOutput = camMonoRight->requestOutput(
            std::make_pair(640, 400),
            dai::ImgFrame::Type::GRAY8,
            dai::ImgResizeMode::CROP,
            config.fps
        );

        Logger::info("Mono L/R: 640x400 GRAY8 @ ", config.fps, " FPS");

        // ─────────────────────────────────────────────────────────────
        // 4. Sync Node - Synchronize all three streams
        // ─────────────────────────────────────────────────────────────
        auto sync = pipeline_->create<dai::node::Sync>();
        sync->setSyncThreshold(std::chrono::milliseconds(33));

        rgbPreview->link(sync->inputs["rgb"]);
        monoLeftOutput->link(sync->inputs["monoLeft"]);
        monoRightOutput->link(sync->inputs["monoRight"]);

        auto syncQueue = sync->out.createOutputQueue(8, false);
        queues_["sync"] = syncQueue;

        Logger::info("Sync Node: 33ms threshold (RGB + Mono L/R)");

    } else {
        // ─────────────────────────────────────────────────────────────
        // RGB-ONLY MODE (Phase 1-2, no stereo)
        // ─────────────────────────────────────────────────────────────
        auto rgbQueue = rgbPreview->createOutputQueue(8, false);
        queues_["rgb"] = rgbQueue;

        Logger::info("RGB-Only Mode: No Stereo (enable with config.enableStereo)");
    }

    Logger::info("═══════════════════════════════════════════════════════════");
    Logger::info("V3 SENSOR-ONLY PIPELINE CREATED");
    Logger::info("  RGB:        ", config.previewWidth, "x", config.previewHeight, " NV12 @ ", config.fps, " FPS");
    if (config.enableStereo) {
        Logger::info("  Mono L/R:   640x400 GRAY8 @ ", config.fps, " FPS");
        Logger::info("  Sync:       33ms threshold");
    } else {
        Logger::info("  Stereo:     DISABLED (Phase 1-2)");
    }
    Logger::info("  NNs:        DISABLED (will run on Jetson via TensorRT)");
    Logger::info("═══════════════════════════════════════════════════════════");

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
        Logger::info("Stopping pipeline and closing device (PoE reconnect fix)...");
         try {
            // 1. Stop pipeline first (releases resources on device)
            if (pipeline_) {
                Logger::debug("Stopping pipeline...");
                // Pipeline destructor will handle cleanup
                pipeline_.reset();
            }

            // 2. Clear queues (releases XLink buffers)
            Logger::debug("Clearing queues...");
            queues_.clear();

            // 3. Close device connection (XLink cleanup)
            if (!device_->isClosed()) {
                Logger::debug("Closing device connection...");
                device_->close();
                Logger::debug("Device connection closed.");
            }

            // 4. Reset device pointer (releases memory)
            Logger::debug("Resetting device pointer...");
            device_.reset();

            // 5. CRITICAL: Wait for PoE network stack to fully reset
            // PoE devices need more time than USB:
            // - TCP connection teardown
            // - ARP cache flush
            // - Device firmware reset
            // INCREASED from 2s to 5s to fix reconnect issues
            Logger::info("Waiting 5 seconds for PoE network stack reset...");
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));

            Logger::info("Device shutdown complete. Ready for reconnect.");
        } catch (const std::exception& e) {
            Logger::error("Error during device shutdown: ", e.what());
            // Force reset even on error
            pipeline_.reset();
            queues_.clear();
            device_.reset();
            // Still wait (full 5s) to give device time to recover
            Logger::warn("Forcing 5s wait after error...");
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
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

