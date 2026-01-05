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
        // Note: InputLoop currently copies whatever comes out of 'rgb' queue.
        // If we want full res for processing, we should request full res output.
        // For now, let's stick to preview size for visualization/debug,
        // but for HandTracking we usually need the NN input size.
        // Let's assume we want the preview stream.
        config.previewWidth = 640;
        config.previewHeight = 360;
        config.nnPath = "models/hand_landmark_full_sh4.blob";

        // 3. Init & Start Pipeline
        pipelineManager->init(config);
        pipelineManager->start();

        // 4. Start Input Loop
        core::InputLoop inputLoop(pipelineManager, framePool, processingQueue);
        inputLoop.start();

        // 5. Start OSC Sender
        // TODO: Load host/port from config
        net::OscSender oscSender(oscQueue, "127.0.0.1", "9000");
        oscSender.start();

        // 6. Start Processing Loop
        core::ProcessingLoop processingLoop(processingQueue, framePool, oscQueue);
        processingLoop.start();

        core::Logger::info("Service running. Press Ctrl+C to exit.");

        // Main loop (Orchestrator)
        while (g_running) {
            // Monitor health, print stats
            // TODO: Implement PerformanceMonitor here

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // Shutdown
        processingLoop.stop();
        oscSender.stop();
        inputLoop.stop();
        pipelineManager->stop();

    } catch (const std::exception& e) {
        core::Logger::error("Fatal error: ", e.what());
        return 1;
    }

    core::Logger::info("Service stopped cleanly.");
    return 0;
}
