#include "core/SystemMonitor.hpp"
#include "core/Logger.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>

namespace core {

std::string SystemMonitor::readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    std::string content;
    std::getline(file, content);
    return content;
}

int SystemMonitor::readIntFromFile(const std::string& path) {
    std::string content = readFile(path);
    if (content.empty()) return -1;
    try {
        return std::stoi(content);
    } catch (...) {
        return -1;
    }
}

SystemPerformance SystemMonitor::getPerformanceStatus() {
    SystemPerformance perf;

    // Read nvpmodel status (Power Mode)
    FILE* pipe = popen("nvpmodel -q 2>/dev/null | head -n1", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            std::string mode(buffer);
            if (mode.find("0") != std::string::npos || mode.find("MAXN") != std::string::npos) {
                perf.powerMode = "MAXN";
            } else if (mode.find("15W") != std::string::npos) {
                perf.powerMode = "15W";
            } else if (mode.find("10W") != std::string::npos) {
                perf.powerMode = "10W";
            } else {
                perf.powerMode = "Unknown";
            }
        }
        pclose(pipe);
    }

    // Read CPU frequency (first core)
    perf.cpuFreqMHz = readIntFromFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    if (perf.cpuFreqMHz > 0) perf.cpuFreqMHz /= 1000; // Convert kHz to MHz

    perf.cpuMaxFreqMHz = readIntFromFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq");
    if (perf.cpuMaxFreqMHz > 0) perf.cpuMaxFreqMHz /= 1000;

    // GPU frequency - Jetson Orin Nano: 17000000.gpu
    perf.gpuFreqMHz = readIntFromFile("/sys/class/devfreq/17000000.gpu/cur_freq");
    if (perf.gpuFreqMHz > 0) perf.gpuFreqMHz /= 1000000;

    perf.gpuMaxFreqMHz = readIntFromFile("/sys/class/devfreq/17000000.gpu/max_freq");
    if (perf.gpuMaxFreqMHz > 0) perf.gpuMaxFreqMHz /= 1000000;

    // Memory frequency - not available
    perf.emcFreqMHz = -1;

    // Temperature
    perf.temperature = readIntFromFile("/sys/devices/virtual/thermal/thermal_zone0/temp");
    if (perf.temperature > 0) perf.temperature /= 1000;

    // Check if at max performance - just check power mode
    perf.isMaxPerformance = (perf.powerMode == "MAXN");

    return perf;
}

std::string SystemMonitor::getPerformanceSummary() {
    auto perf = getPerformanceStatus();
    std::ostringstream ss;
    ss << perf.powerMode
       << " CPU:" << perf.cpuFreqMHz << "MHz"
       << " GPU:" << perf.gpuFreqMHz << "MHz"
       << " " << perf.temperature << "C";
    return ss.str();
}

bool SystemMonitor::ensureMaxPerformance() {
    auto perf = getPerformanceStatus();

    if (perf.isMaxPerformance) {
        Logger::info("System at maximum performance: ", getPerformanceSummary());
        return true;
    }

    Logger::warn("System NOT at maximum performance!");
    Logger::warn("Current: ", getPerformanceSummary());
    Logger::warn("Setting MAXN mode...");

    // Use standard NVIDIA commands directly
    system("sudo nvpmodel -m 0 2>/dev/null");
    system("sudo jetson_clocks 2>/dev/null");

    // Wait for changes to take effect
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto newPerf = getPerformanceStatus();
    Logger::info("After optimization: ", getPerformanceSummary());

    if (newPerf.powerMode != "MAXN") {
        Logger::warn("Could not set MAXN mode. Run manually:");
        Logger::warn("  sudo nvpmodel -m 0 && sudo jetson_clocks");
    }

    return newPerf.isMaxPerformance;
}

} // namespace core

