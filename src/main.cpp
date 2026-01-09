#include "core/PipelineManager.hpp"
#include "core/Logger.hpp"
#include "core/Types.hpp"
#include "core/InputLoop.hpp"
#include "core/ProcessingLoop.hpp"
#include "core/SystemMonitor.hpp"
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

    // Log System Performance (informational only)
    core::Logger::info("System: ", core::SystemMonitor::getPerformanceSummary());
    auto perf = core::SystemMonitor::getPerformanceStatus();
    if (perf.powerMode != "MAXN") {
        core::Logger::warn("Not in MAXN mode! Run: sudo nvpmodel -m 0 && sudo jetson_clocks");
    }

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

            // 2. Configuration - V3 Sensor-Only Pipeline
            core::PipelineManager::Config config;
            config.fps = 30.0f;  // V3: 30 FPS for stable operation
            config.ispScaleNum = 1;
            config.ispScaleDenom = 3; // 1080p → 360p
            config.previewWidth = 640;   // RGB Preview width
            config.previewHeight = 360;  // RGB Preview height
            config.monoWidth = 640;      // Mono L/R width (THE_400_P)
            config.monoHeight = 400;     // Mono L/R height (THE_400_P)
            config.enableStereo = false; // V3 Phase 1-2: RGB-only (enable in Phase 3)
            config.nnPath = "";  // V3: NNs disabled on OAK-D, run on Jetson
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

            // Explicitly release shared pointers to force destruction of DepthAI objects
            // before the delay and next iteration.
            // This ensures the XLink connection is fully closed by the destructor.
            // (inputLoop holds a shared_ptr to pipelineManager)
            // We are at the end of the scope so they would be destroyed anyway,
            // but doing it before the sleep is safer.
            // Since they are stack objects here, we can't 'reset' them easily if they aren't pointers
            // except by exiting the scope.
            // But main variables are pointers? No...
            // pipelineManager is shared_ptr. frames/queues are shared_ptr.
            // inputLoop, oscSender, processingLoop are stack objects.

            // To be 100% sure, we can rely on the scope exit, but let's make sure
            // pipelineManager->stop() did the heavy lifting (clearing device_).

            if (!g_running) {
                break; // Exit outer loop if user requested shutdown
            }

            core::Logger::info("Restarting in 5 seconds...");
            std::this_thread::sleep_for(std::chrono::seconds(5));

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
