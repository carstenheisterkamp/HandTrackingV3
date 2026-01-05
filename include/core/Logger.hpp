#pragma once

#include <iostream>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace core {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

/**
 * Thread-safe Logger utility.
 * Note: Avoid using in the absolute hot-path (inference loop) unless necessary,
 * as I/O can block.
 */
class Logger {
public:
    static void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::cout << "[" << std::put_time(std::localtime(&time), "%H:%M:%S")
                  << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";

        switch (level) {
            case LogLevel::DEBUG: std::cout << "\033[36m[DEBUG]\033[0m "; break; // Cyan
            case LogLevel::INFO:  std::cout << "\033[32m[INFO] \033[0m "; break; // Green
            case LogLevel::WARN:  std::cout << "\033[33m[WARN] \033[0m "; break; // Yellow
            case LogLevel::ERROR: std::cout << "\033[31m[ERROR]\033[0m "; break; // Red
        }

        std::cout << message << std::endl;
    }

    // Helper for formatted logging
    template<typename... Args>
    static void debug(Args... args) {
        std::stringstream ss;
        (ss << ... << args);
        log(LogLevel::DEBUG, ss.str());
    }

    template<typename... Args>
    static void info(Args... args) {
        std::stringstream ss;
        (ss << ... << args);
        log(LogLevel::INFO, ss.str());
    }

    template<typename... Args>
    static void warn(Args... args) {
        std::stringstream ss;
        (ss << ... << args);
        log(LogLevel::WARN, ss.str());
    }

    template<typename... Args>
    static void error(Args... args) {
        std::stringstream ss;
        (ss << ... << args);
        log(LogLevel::ERROR, ss.str());
    }

private:
    static std::mutex mutex_;
};

} // namespace core

