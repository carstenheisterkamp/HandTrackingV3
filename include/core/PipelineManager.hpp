#pragma once

#include <depthai/depthai.hpp>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <map>

namespace dai {
    class MessageQueue;
}

namespace core {

/**
 * Manages the DepthAI v3 Pipeline and Device connection.
 * Handles OAK-D Pro PoE initialization, pipeline construction, and runtime updates.
 */
class PipelineManager {
public:
    struct Config {
        float fps = 30.0f;
        int ispScaleNum = 1;
        int ispScaleDenom = 3; // 1080p -> 360p (example) or similar scaling
        int previewWidth = 640;
        int previewHeight = 360;
        std::string nnPath; // Path to the neural network blob
        std::string deviceIp; // IP address for PoE devices (e.g., "169.254.1.222")
    };

    PipelineManager();
    ~PipelineManager();

    // Non-copyable
    PipelineManager(const PipelineManager&) = delete;
    PipelineManager& operator=(const PipelineManager&) = delete;

    /**
     * Initializes the device and pipeline.
     * Throws std::runtime_error on failure.
     */
    void init(const Config& config);

    /**
     * Starts the pipeline.
     */
    void start();

    /**
     * Stops the pipeline and closes the device connection.
     */
    void stop();

    /**
     * Updates pipeline parameters at runtime (e.g. exposure, FPS).
     * Uses DepthAI v3 pipeline update mechanism.
     */
    void update(const std::string& nodeName, const std::string& param, float value);

    /**
     * Returns the output queue for a specific stream.
     */
    std::shared_ptr<dai::MessageQueue> getOutputQueue(const std::string& name, int maxSize = 4, bool blocking = false);

    /**
     * Returns the underlying Device pointer (useful for low-level access if needed).
     */
    std::shared_ptr<dai::Device> getDevice() const { return device_; }

private:
    std::shared_ptr<dai::Device> device_;
    std::unique_ptr<dai::Pipeline> pipeline_;
    Config currentConfig_;

    // Store queues created during pipeline construction
    std::map<std::string, std::shared_ptr<dai::MessageQueue>> queues_;

    void createPipeline(const Config& config);
};

} // namespace core

