#include "core/PipelineManager.hpp"
#include "core/Logger.hpp"
#include "core/Types.hpp"
#include "core/InputLoop.hpp"
#include "core/ProcessingLoop.hpp"
#include "net/OscSender.hpp"
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

// Global flag for shutdown
std::atomic<bool> g_running{true};

void signalHandler(int signum) {
    core::Logger::info("Interrupt signal (", signum, ") received. Shutting down...");
    g_running = false;
}

int main() {
    // Register signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    core::Logger::info("Starting HandTrackingService V3...");

    // Configuration Constants
    const std::string OSC_HOST = "127.0.0.1";
    const std::string OSC_PORT = "9000";

        // Outer Loop for auto-restart
    while (g_running) {
        try {
            // 1. Infrastructure
            auto pipelineManager = std::make_shared<core::PipelineManager>();

            // Use types from Types.hpp
            auto framePool = std::make_shared<core::AppFramePool>();
            auto processingQueue = std::make_shared<core::AppProcessingQueue>();
            auto oscQueue = std::make_shared<core::OscQueue>();

            // 2. Configuration
            core::PipelineManager::Config config;
            config.fps = 30.0f;
            config.ispScaleNum = 1;
            config.ispScaleDenom = 3; // 1080p -> 360p (Preview)
            config.previewWidth = 640;
            config.previewHeight = 360;
            config.nnPath = "models/hand_landmark_full_sh4.blob";
            config.deviceIp = "169.254.1.222"; // OAK-D Pro PoE IP Address

            // 3. Init & Start Pipeline
            pipelineManager->init(config);
            pipelineManager->start();

            // 4. Start Input Loop
            core::InputLoop inputLoop(pipelineManager, framePool, processingQueue);
            inputLoop.start();

            // 5. Start OSC Sender
            net::OscSender oscSender(oscQueue, OSC_HOST, OSC_PORT);
            oscSender.start();

            // 6. Start Processing Loop
            core::ProcessingLoop processingLoop(processingQueue, framePool, oscQueue);
            processingLoop.start();

            core::Logger::info("Service running. Press Ctrl+C to exit.");

            // Main loop (Orchestrator)
            while (g_running) {
                // Check for InputLoop errors (e.g. Camera disconnected)
                if (inputLoop.hasError()) {
                    core::Logger::warn("InputLoop reported critical error. Restarting service...");
                    break; // Break inner loop to trigger restart
                }

                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            // Shutdown: Stop explicitly to ensure clean cleanup order
            core::Logger::info("Stopping modules...");
            inputLoop.stop();
            processingLoop.stop();
            oscSender.stop();
            pipelineManager->stop();

            if (!g_running) {
                break; // Exit outer loop if user requested shutdown
            }

            core::Logger::info("Restarting in 2 seconds...");
            std::this_thread::sleep_for(std::chrono::seconds(2));

        } catch (const std::exception& e) {
            core::Logger::error("Fatal error in service loop: ", e.what());
            if (g_running) {
                core::Logger::info("Retrying in 5 seconds...");
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }
    }

    core::Logger::info("Service stopped cleanly.");
    return 0;
}
