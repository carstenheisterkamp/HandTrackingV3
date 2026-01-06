#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable> // Required for std::condition_variable
#include <atomic>
#include <opencv2/opencv.hpp>
#include <netinet/in.h>

namespace net {

class MjpegServer {
public:
    MjpegServer(int port = 8080);
    ~MjpegServer();

    void start();
    void stop();

    /**
     * Update the stream with a new frame.
     * The frame will be encoded to JPEG and sent to all connected clients.
     */
    void update(const cv::Mat& frame);

    bool hasClients();

private:
    void serverLoop();
    void handleClient(int clientSocket);
    void cleanClients();

    int _serverSocket = -1;
    int _port;
    std::atomic<bool> _running;
    std::thread _serverThread;

    struct Client {
        int socket;
        std::thread thread;
        std::atomic<bool> active;
    };

    std::vector<std::shared_ptr<Client>> _clients;
    std::mutex _clientsMutex;

    // Latest frame buffer
    std::vector<uchar> _currentJpeg;
    std::mutex _frameMutex;
    std::condition_variable _frameCv;
};

} // namespace net
