#include "net/MjpegServer.hpp"
#include "core/Logger.hpp"
#include <sys/socket.h>
#include <unistd.h>
#include <iostream>
#include <sstream>

namespace net {

MjpegServer::MjpegServer(int port) : _port(port), _running(false) {
}

MjpegServer::~MjpegServer() {
    stop();
}

void MjpegServer::start() {
    if (_running) return;

    _serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (_serverSocket < 0) {
        core::Logger::error("MjpegServer: Failed to create socket");
        return;
    }

    int opt = 1;
    setsockopt(_serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(_port);

    if (bind(_serverSocket, (struct sockaddr*)&address, sizeof(address)) < 0) {
        core::Logger::error("MjpegServer: Failed to bind to port ", _port);
        close(_serverSocket);
        return;
    }

    if (listen(_serverSocket, 5) < 0) {
        core::Logger::error("MjpegServer: Failed to listen");
        close(_serverSocket);
        return;
    }

    _running = true;
    _serverThread = std::thread(&MjpegServer::serverLoop, this);
    core::Logger::info("MjpegServer started on port ", _port);
}

void MjpegServer::stop() {
    if (!_running) return;
    _running = false;

    // Close server socket to unblock accept()
    if (_serverSocket >= 0) {
        close(_serverSocket);
        _serverSocket = -1;
    }

    if (_serverThread.joinable()) {
        _serverThread.join();
    }

    // Stop all clients
    {
        std::lock_guard<std::mutex> lock(_clientsMutex);
        for (auto& client : _clients) {
            client->active = false;
            if (client->socket >= 0) {
                close(client->socket);
            }
            if (client->thread.joinable()) {
                client->thread.join();
            }
        }
        _clients.clear();
    }

    // Wake up any waiting threads
    _frameCv.notify_all();

    core::Logger::info("MjpegServer stopped.");
}

void MjpegServer::update(const cv::Mat& frame) {
    if (!_running || frame.empty()) return;

    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};

    try {
        cv::imencode(".jpg", frame, buf, params);
    } catch (const cv::Exception& e) {
        core::Logger::error("MjpegServer: JPEG encoding failed: ", e.what());
        return;
    }

    {
        std::lock_guard<std::mutex> lock(_frameMutex);
        _currentJpeg = std::move(buf);
    }
    _frameCv.notify_all();

    cleanClients();
}

bool MjpegServer::hasClients() {
    std::lock_guard<std::mutex> lock(_clientsMutex);
    return !_clients.empty();
}

void MjpegServer::serverLoop() {
    while (_running) {
        sockaddr_in clientAddr;
        socklen_t clientLen = sizeof(clientAddr);
        int clientSocket = accept(_serverSocket, (struct sockaddr*)&clientAddr, &clientLen);

        if (clientSocket < 0) {
            if (_running) {
                core::Logger::warn("MjpegServer: Accept failed");
            }
            continue;
        }

        core::Logger::info("MjpegServer: New client connected");

        auto client = std::make_shared<Client>();
        client->socket = clientSocket;
        client->active = true;
        client->thread = std::thread(&MjpegServer::handleClient, this, clientSocket);

        // Detach thread or keep it? Better to keep track.
        // But handleClient is a member function, so we need to be careful about lifetime.
        // We'll store the client object.

        std::lock_guard<std::mutex> lock(_clientsMutex);
        _clients.push_back(client);
    }
}

void MjpegServer::handleClient(int clientSocket) {
    // Send HTTP Header
    std::string header = "HTTP/1.1 200 OK\r\n"
                         "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                         "\r\n";

    if (send(clientSocket, header.c_str(), header.size(), 0) < 0) {
        close(clientSocket);
        return;
    }

    while (_running) {
        std::vector<uchar> jpegData;
        {
            std::unique_lock<std::mutex> lock(_frameMutex);
            _frameCv.wait(lock, [this]{ return !_currentJpeg.empty() || !_running; });

            if (!_running) break;
            jpegData = _currentJpeg;
        }

        if (jpegData.empty()) continue;

        std::ostringstream oss;
        oss << "--frame\r\n"
            << "Content-Type: image/jpeg\r\n"
            << "Content-Length: " << jpegData.size() << "\r\n"
            << "\r\n";

        std::string boundary = oss.str();

        // Send Boundary
        if (send(clientSocket, boundary.c_str(), boundary.size(), MSG_NOSIGNAL) < 0) break;

        // Send JPEG
        if (send(clientSocket, jpegData.data(), jpegData.size(), MSG_NOSIGNAL) < 0) break;

        // Send Newline
        if (send(clientSocket, "\r\n", 2, MSG_NOSIGNAL) < 0) break;
    }

    close(clientSocket);

    // Mark as inactive
    std::lock_guard<std::mutex> lock(_clientsMutex);
    for (auto& client : _clients) {
        if (client->socket == clientSocket) {
            client->active = false;
            break;
        }
    }
}

void MjpegServer::cleanClients() {
    std::lock_guard<std::mutex> lock(_clientsMutex);
    auto it = _clients.begin();
    while (it != _clients.end()) {
        if (!(*it)->active) {
            if ((*it)->thread.joinable()) {
                (*it)->thread.join();
            }
            it = _clients.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace net

