#pragma once

#include <string>
#include <map>

namespace core {

struct SystemPerformance {
    std::string powerMode;         // MAXN, 15W, etc.
    int cpuFreqMHz;                // Current CPU frequency
    int cpuMaxFreqMHz;             // Maximum CPU frequency
    int gpuFreqMHz;                // Current GPU frequency
    int gpuMaxFreqMHz;             // Maximum GPU frequency
    int emcFreqMHz;                // Current EMC (Memory) frequency
    int temperature;               // GPU temperature in Celsius
    bool isMaxPerformance;         // True if running at max
};

class SystemMonitor {
public:
    static SystemPerformance getPerformanceStatus();
    static std::string getPerformanceSummary();
    static bool ensureMaxPerformance();  // Returns true if already at max or successfully set

private:
    static std::string readFile(const std::string& path);
    static int readIntFromFile(const std::string& path);
};

} // namespace core

