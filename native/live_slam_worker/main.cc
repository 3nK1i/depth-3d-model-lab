#include <stella_vslam/config.h>
#include <stella_vslam/data/keyframe.h>
#include <stella_vslam/data/landmark.h>
#include <stella_vslam/publish/map_publisher.h>
#include <stella_vslam/system.h>
#include <stella_vslam/3rd/json/include/nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef EIA_STELLA_VERSION
#define EIA_STELLA_VERSION "unknown"
#endif
#ifndef EIA_STELLA_COMMIT
#define EIA_STELLA_COMMIT "unknown"
#endif

namespace {

using json = nlohmann::json;
constexpr int kProtocolVersion = 1;
constexpr std::size_t kMaxMessageBytes = 1024 * 1024;
constexpr std::size_t kHeaderBytes = sizeof(std::uint64_t);
constexpr const char* kWorkerVersion = "phase8.8c4-worker-v1";
constexpr const char* kPoseInputConvention = "T_camera_from_reconstruction";
constexpr const char* kPoseOutputConvention = "T_reconstruction_from_camera";
constexpr const char* kReconstructionFrame = "arbitrary_gauge_live_reconstruction";
constexpr std::size_t kMaxSparseLandmarkSamples = 4096;
constexpr std::size_t kMaxLiveKeyframePoseSamples = 256;
constexpr std::uint64_t kSparseMapPublishIntervalFrames = 10;

struct Options {
    std::string socket_path;
    std::string session_id;
    std::string vocab_path;
    std::string config_path;
    std::string backend_id = "stella-vslam-live";
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string errno_message(const std::string& prefix) {
    return prefix + ": " + std::strerror(errno);
}

void print_help(const char* argv0) {
    std::cout
        << "Depth & 3D Model Lab isolated stella_vslam live worker\n"
        << "Usage: " << argv0
        << " --socket PATH --session-id UUID --vocab FILE --config FILE [--backend-id ID]\n"
        << "The worker accepts frames only through the Phase 8.4 shared-memory ring.\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        }
        if (arg == "--version") {
            std::cout << "eia_stella_live_worker " << kWorkerVersion
                      << " stella_vslam=" << EIA_STELLA_VERSION
                      << " commit=" << EIA_STELLA_COMMIT << "\n";
            std::exit(0);
        }
        if (i + 1 >= argc) {
            fail("Missing value for argument: " + arg);
        }
        const std::string value(argv[++i]);
        if (arg == "--socket") options.socket_path = value;
        else if (arg == "--session-id") options.session_id = value;
        else if (arg == "--vocab") options.vocab_path = value;
        else if (arg == "--config") options.config_path = value;
        else if (arg == "--backend-id") options.backend_id = value;
        else fail("Unknown argument: " + arg);
    }
    if (options.socket_path.empty() || options.session_id.empty() ||
        options.vocab_path.empty() || options.config_path.empty() ||
        options.backend_id.empty()) {
        fail("--socket, --session-id, --vocab, --config and backend id are required");
    }
    if (options.socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        fail("Unix socket path is too long");
    }
    return options;
}

class ScopedFd {
public:
    ScopedFd() = default;
    explicit ScopedFd(int fd) : fd_(fd) {}
    ~ScopedFd() { reset(); }
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    ScopedFd(ScopedFd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    ScopedFd& operator=(ScopedFd&& other) noexcept {
        if (this != &other) {
            reset();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }
    int get() const { return fd_; }
    explicit operator bool() const { return fd_ >= 0; }
    void reset(int value = -1) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = value;
    }
private:
    int fd_ = -1;
};

bool safe_remove_socket_path(const std::string& path) {
    struct stat st {};
    if (::lstat(path.c_str(), &st) != 0) {
        return errno == ENOENT;
    }
    if (!S_ISSOCK(st.st_mode) || st.st_uid != ::geteuid()) {
        return false;
    }
    return ::unlink(path.c_str()) == 0;
}

class UnixServer {
public:
    explicit UnixServer(const std::string& path) : path_(path) {
        ::umask(0077);
        if (!safe_remove_socket_path(path_)) {
            fail("Refusing to remove non-owned/non-socket Unix endpoint: " + path_);
        }
        ScopedFd server(::socket(AF_UNIX, SOCK_STREAM, 0));
        if (!server) fail(errno_message("socket(AF_UNIX) failed"));
        sockaddr_un address {};
        address.sun_family = AF_UNIX;
        std::strncpy(address.sun_path, path_.c_str(), sizeof(address.sun_path) - 1);
        if (::bind(server.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
            fail(errno_message("bind(AF_UNIX) failed"));
        }
        if (::chmod(path_.c_str(), 0600) != 0) {
            fail(errno_message("chmod Unix socket failed"));
        }
        if (::listen(server.get(), 1) != 0) {
            fail(errno_message("listen(AF_UNIX) failed"));
        }
        server_ = std::move(server);
    }

    ~UnixServer() {
        client_.reset();
        server_.reset();
        if (!path_.empty()) ::unlink(path_.c_str());
    }

    void accept_one() {
        const int fd = ::accept(server_.get(), nullptr, nullptr);
        if (fd < 0) fail(errno_message("accept(AF_UNIX) failed"));
        client_.reset(fd);
    }

    int client_fd() const {
        if (!client_) fail("No live SLAM client is connected");
        return client_.get();
    }

private:
    std::string path_;
    ScopedFd server_;
    ScopedFd client_;
};

void send_all(int fd, const std::string& value) {
    std::size_t offset = 0;
    while (offset < value.size()) {
        const ssize_t sent = ::send(fd, value.data() + offset, value.size() - offset, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            fail(errno_message("send live SLAM protocol failed"));
        }
        if (sent == 0) fail("Live SLAM protocol socket closed while sending");
        offset += static_cast<std::size_t>(sent);
    }
}

class JsonLineChannel {
public:
    JsonLineChannel(int fd, std::string session_id)
        : fd_(fd), session_id_(std::move(session_id)) {}

    void send_message(const std::string& type, const json& payload) {
        json message = {
            {"protocol_version", kProtocolVersion},
            {"message_type", type},
            {"session_id", session_id_},
            {"sequence_number", worker_sequence_++},
            {"payload", payload},
        };
        std::string encoded = message.dump();
        encoded.push_back('\n');
        if (encoded.size() > kMaxMessageBytes) fail("Worker attempted oversized JSON response");
        send_all(fd_, encoded);
    }

    json receive_message() {
        for (;;) {
            const auto newline = buffer_.find('\n');
            if (newline != std::string::npos) {
                std::string line = buffer_.substr(0, newline);
                buffer_.erase(0, newline + 1);
                if (line.empty()) continue;
                json message;
                try {
                    message = json::parse(line);
                } catch (const std::exception&) {
                    fail("Malformed JSON from live SLAM client");
                }
                validate_message(message);
                return message;
            }
            if (buffer_.size() > kMaxMessageBytes) fail("Oversized JSON from live SLAM client");
            char chunk[65536];
            const ssize_t received = ::recv(fd_, chunk, sizeof(chunk), 0);
            if (received < 0) {
                if (errno == EINTR) continue;
                fail(errno_message("recv live SLAM protocol failed"));
            }
            if (received == 0) fail("Live SLAM client closed the control socket");
            buffer_.append(chunk, static_cast<std::size_t>(received));
        }
    }

private:
    void validate_message(const json& message) {
        if (!message.is_object()) fail("Live SLAM protocol root must be an object");
        if (message.value("protocol_version", 0) != kProtocolVersion) {
            fail("Live SLAM protocol version mismatch");
        }
        if (message.value("session_id", std::string()) != session_id_) {
            fail("Live SLAM protocol session mismatch");
        }
        if (!message.contains("sequence_number") || !message["sequence_number"].is_number_unsigned()) {
            fail("Live SLAM client sequence_number must be non-negative integer");
        }
        const auto sequence = message["sequence_number"].get<std::uint64_t>();
        if (has_client_sequence_ && sequence <= last_client_sequence_) {
            fail("Live SLAM client sequence_number is not strictly increasing");
        }
        last_client_sequence_ = sequence;
        has_client_sequence_ = true;
        if (!message.contains("message_type") || !message["message_type"].is_string()) {
            fail("Live SLAM message_type must be a string");
        }
        if (!message.contains("payload") || !message["payload"].is_object()) {
            fail("Live SLAM payload must be an object");
        }
    }

    int fd_;
    std::string session_id_;
    std::string buffer_;
    std::uint64_t worker_sequence_ = 0;
    std::uint64_t last_client_sequence_ = 0;
    bool has_client_sequence_ = false;
};

struct RingSpec {
    std::string name;
    std::size_t slot_count = 0;
    std::size_t max_payload_bytes = 0;
    std::size_t slot_stride_bytes = 0;
    std::size_t total_bytes = 0;
};

std::string normalized_shm_name(const std::string& value) {
    if (value.empty()) fail("Shared-memory name is empty");
    std::string raw = value;
    if (!raw.empty() && raw.front() == '/') raw.erase(raw.begin());
    if (raw.empty()) fail("Shared-memory name is empty");
    for (const char c : raw) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        if (!ok) fail("Shared-memory name contains unsupported characters");
    }
    return "/" + raw;
}

class ReadOnlyRing {
public:
    ReadOnlyRing() = default;
    ~ReadOnlyRing() { close(); }
    ReadOnlyRing(const ReadOnlyRing&) = delete;
    ReadOnlyRing& operator=(const ReadOnlyRing&) = delete;

    void open(const RingSpec& spec) {
        close();
        if (spec.slot_count == 0 || spec.max_payload_bytes == 0 ||
            spec.slot_stride_bytes < kHeaderBytes + spec.max_payload_bytes ||
            spec.total_bytes != spec.slot_count * spec.slot_stride_bytes) {
            fail("Invalid shared-memory ring geometry");
        }
        const std::string shm_name = normalized_shm_name(spec.name);
        fd_.reset(::shm_open(shm_name.c_str(), O_RDONLY, 0));
        if (!fd_) fail(errno_message("shm_open failed"));
        struct stat st {};
        if (::fstat(fd_.get(), &st) != 0) fail(errno_message("fstat shared memory failed"));
        if (st.st_size < 0 || static_cast<std::size_t>(st.st_size) < spec.total_bytes) {
            fail("Shared-memory object is smaller than declared ring");
        }
        void* mapping = ::mmap(nullptr, spec.total_bytes, PROT_READ, MAP_SHARED, fd_.get(), 0);
        if (mapping == MAP_FAILED) fail(errno_message("mmap shared memory failed"));
        data_ = static_cast<const std::uint8_t*>(mapping);
        mapped_bytes_ = spec.total_bytes;
        spec_ = spec;
    }

    void close() {
        if (data_ != nullptr) {
            ::munmap(const_cast<std::uint8_t*>(data_), mapped_bytes_);
            data_ = nullptr;
            mapped_bytes_ = 0;
        }
        fd_.reset();
        spec_ = RingSpec{};
    }

    std::vector<std::uint8_t> read_frame(
        std::size_t slot_index,
        std::uint64_t generation,
        std::size_t byte_length) const {
        if (data_ == nullptr) fail("Shared-memory ring is not open");
        if (slot_index >= spec_.slot_count) fail("Shared-memory frame slot is outside ring");
        if (generation == 0) fail("Shared-memory generation must be positive");
        if (byte_length == 0 || byte_length > spec_.max_payload_bytes) {
            fail("Shared-memory frame byte length is invalid");
        }
        const std::size_t base = slot_index * spec_.slot_stride_bytes;
        const auto* token_ptr = reinterpret_cast<const std::uint64_t*>(data_ + base);
        const std::uint64_t expected = generation << 1U;
        const std::uint64_t before = __atomic_load_n(token_ptr, __ATOMIC_ACQUIRE);
        if ((before & 1U) != 0U || before != expected) {
            throw std::runtime_error("EIA_STALE_FRAME:shared-memory generation was overwritten before read");
        }
        std::vector<std::uint8_t> bytes(byte_length);
        std::memcpy(bytes.data(), data_ + base + kHeaderBytes, byte_length);
        std::atomic_thread_fence(std::memory_order_acquire);
        const std::uint64_t after = __atomic_load_n(token_ptr, __ATOMIC_ACQUIRE);
        if ((after & 1U) != 0U || after != expected) {
            throw std::runtime_error("EIA_STALE_FRAME:shared-memory generation changed during read");
        }
        return bytes;
    }

private:
    ScopedFd fd_;
    const std::uint8_t* data_ = nullptr;
    std::size_t mapped_bytes_ = 0;
    RingSpec spec_;
};

RingSpec parse_ring_spec(const json& payload) {
    if (!payload.is_object()) fail("frame_ring must be an object");
    if (payload.value("schema_version", 0) != 1) fail("Unsupported frame_ring schema version");
    RingSpec spec;
    spec.name = payload.at("shared_memory_name").get<std::string>();
    spec.slot_count = payload.at("slot_count").get<std::size_t>();
    spec.max_payload_bytes = payload.at("max_payload_bytes").get<std::size_t>();
    spec.slot_stride_bytes = payload.at("slot_stride_bytes").get<std::size_t>();
    spec.total_bytes = payload.at("total_bytes").get<std::size_t>();
    return spec;
}

json matrix_to_json(const Eigen::Matrix4d& matrix) {
    json rows = json::array();
    for (int row = 0; row < 4; ++row) {
        json values = json::array();
        for (int col = 0; col < 4; ++col) values.push_back(matrix(row, col));
        rows.push_back(values);
    }
    return rows;
}

class StellaSession {
public:
    StellaSession(const Options& options, JsonLineChannel& channel)
        : options_(options), channel_(channel) {}

    ~StellaSession() { stop_noexcept(); }

    void start(const json& payload) {
        if (running_) fail("stella_vslam session is already running");
        if (!payload.contains("request") || !payload["request"].is_object()) {
            fail("START payload.request is required");
        }
        const json& request = payload["request"];
        if (request.value("schema_version", 0) != 1) fail("Unsupported live SLAM request schema");
        if (request.value("session_id", std::string()) != options_.session_id) {
            fail("START request session mismatch");
        }
        if (request.value("backend_id", std::string()) != options_.backend_id) {
            fail("START backend id mismatch");
        }
        enable_loop_closure_ = request.value("enable_loop_closure", false);
        enable_global_optimization_ = request.value("enable_global_optimization", false);
        if (enable_loop_closure_ != enable_global_optimization_) {
            fail("Qualified stella backend requires loop closure and global optimization together");
        }
        enable_mapping_ = request.value("enable_mapping", true);
        if (!request.contains("camera") || !request["camera"].is_object()) {
            fail("START request.camera is required");
        }
        const json& camera = request["camera"];
        camera_id_ = camera.at("camera_id").get<std::string>();
        if (!camera.contains("camera_frame") || !camera["camera_frame"].is_object()) {
            fail("START request camera_frame is required");
        }
        camera_frame_id_ = camera["camera_frame"].at("frame_id").get<std::string>();
        if (!camera.contains("image_size") || !camera["image_size"].is_object()) {
            fail("START request camera image_size is required");
        }
        width_ = camera["image_size"].at("width").get<int>();
        height_ = camera["image_size"].at("height").get<int>();
        if (width_ <= 0 || height_ <= 0) fail("Calibrated camera dimensions must be positive");

        ring_spec_ = parse_ring_spec(payload.at("frame_ring"));
        const std::size_t expected_frame_bytes = static_cast<std::size_t>(width_) *
                                                 static_cast<std::size_t>(height_) * 3U;
        if (ring_spec_.max_payload_bytes < expected_frame_bytes) {
            fail("Shared-memory ring cannot hold one calibrated BGR frame");
        }
        ring_.open(ring_spec_);

        config_ = std::make_shared<stella_vslam::config>(options_.config_path);
        slam_ = std::make_shared<stella_vslam::system>(config_, options_.vocab_path);
        slam_->startup();
        if (enable_loop_closure_) {
            slam_->enable_loop_detector();
            if (!slam_->loop_detector_is_enabled()) {
                fail("stella_vslam loop detector did not enable for Phase 8.5B session");
            }
        } else {
            // Phase 8.4 default contract retained.  The exact default marker below
            // intentionally remains visible to the backward-boundary source test:
            // {"phase8_global_optimization_feature_enabled", false}
            slam_->disable_loop_detector();
            if (slam_->loop_detector_is_enabled()) {
                fail("stella_vslam loop detector remained enabled after explicit disable");
            }
        }
        if (!enable_mapping_) slam_->disable_mapping_module();
        map_publisher_ = slam_->get_map_publisher();
        if (!map_publisher_) fail("stella_vslam map publisher is unavailable");

        previous_keyframes_ = 0;
        previous_landmarks_ = 0;
        map_revision_ = 0;
        last_sparse_map_revision_ = static_cast<std::uint64_t>(-1);
        optimized_pose_revision_ = 1;
        accepted_loop_closure_requests_ = 0;
        relocalization_count_ = 0;
        last_relocalization_frame_id_.clear();
        previous_optimized_poses_.clear();
        ever_had_pose_ = false;
        previous_lost_ = false;
        running_ = true;
        channel_.send_message(
            "started",
            {
                {"loop_detector_enabled", slam_->loop_detector_is_enabled()},
                {"phase8_global_optimization_feature_enabled", enable_global_optimization_},
                {"internal_global_optimization_thread_may_exist", true},
                {"mapping_enabled", enable_mapping_},
                {"scale_status", "arbitrary"},
                {"unit_to_metre", nullptr},
            });
    }

    void process_frame(const json& payload) {
        if (!running_ || !slam_) fail("FRAME_AVAILABLE received without active stella session");
        if (!payload.contains("frame") || !payload["frame"].is_object()) {
            fail("FRAME_AVAILABLE payload.frame is required");
        }
        const json& frame = payload["frame"];
        const std::string frame_id = frame.at("frame_id").get<std::string>();
        const std::string camera_id = frame.at("camera_id").get<std::string>();
        const std::uint64_t sequence_index = frame.at("sequence_index").get<std::uint64_t>();
        const std::uint64_t timestamp_ns = frame.at("capture_timestamp_ns").get<std::uint64_t>();
        if (camera_id != camera_id_) fail("FRAME_AVAILABLE camera does not match calibrated camera");
        if (!frame.contains("buffer") || !frame["buffer"].is_object()) {
            fail("FRAME_AVAILABLE buffer reference is required");
        }
        const json& buffer = frame["buffer"];
        if (buffer.value("schema_version", 0) != 1 ||
            buffer.value("pixel_format", std::string()) != "bgr8" ||
            buffer.value("dtype", std::string()) != "uint8" ||
            buffer.value("channels", 0) != 3) {
            fail("Unsupported live frame buffer format");
        }
        if (buffer.at("shared_memory_name").get<std::string>() != ring_spec_.name) {
            fail("Frame reference points to another shared-memory ring");
        }
        const json& image_size = buffer.at("image_size");
        const int frame_width = image_size.at("width").get<int>();
        const int frame_height = image_size.at("height").get<int>();
        if (frame_width != width_ || frame_height != height_) {
            fail("Live frame dimensions do not match calibrated camera");
        }
        const std::size_t byte_length = buffer.at("byte_length").get<std::size_t>();
        const std::size_t expected_bytes = static_cast<std::size_t>(width_) *
                                           static_cast<std::size_t>(height_) * 3U;
        if (byte_length != expected_bytes) fail("Live frame byte length does not match BGR dimensions");
        const std::size_t slot_index = buffer.at("slot_index").get<std::size_t>();
        const std::uint64_t generation = buffer.at("generation").get<std::uint64_t>();

        std::vector<std::uint8_t> pixels;
        try {
            pixels = ring_.read_frame(slot_index, generation, byte_length);
        } catch (const std::runtime_error& exc) {
            const std::string message(exc.what());
            if (message.rfind("EIA_STALE_FRAME:", 0) == 0) {
                channel_.send_message(
                    "error",
                    {
                        {"code", "stale_frame"},
                        {"message", message.substr(std::strlen("EIA_STALE_FRAME:"))},
                        {"retryable", true},
                        {"frame_id", frame_id},
                        {"sequence_index", sequence_index},
                    });
                return;
            }
            throw;
        }

        cv::Mat image(height_, width_, CV_8UC3, pixels.data());
        const double timestamp_s = static_cast<double>(timestamp_ns) / 1.0e9;
        const auto started = std::chrono::steady_clock::now();
        const auto cam_pose_cw = slam_->feed_monocular_frame(image, timestamp_s);
        const auto ended = std::chrono::steady_clock::now();
        const auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();

        std::vector<std::shared_ptr<stella_vslam::data::keyframe>> keyframes;
        const unsigned int keyframe_count = map_publisher_->get_keyframes(keyframes);
        std::vector<std::shared_ptr<stella_vslam::data::landmark>> landmarks;
        std::set<std::shared_ptr<stella_vslam::data::landmark>> local_landmarks;
        const unsigned int landmark_count = map_publisher_->get_landmarks(landmarks, local_landmarks);
        if (keyframe_count != previous_keyframes_ || landmark_count != previous_landmarks_) {
            ++map_revision_;
        }
        const bool is_keyframe = cam_pose_cw && keyframe_count > previous_keyframes_;
        previous_keyframes_ = keyframe_count;
        previous_landmarks_ = landmark_count;
        refresh_optimized_pose_revision(keyframes);

        std::string tracking_state;
        std::string reason;
        json pose = nullptr;
        if (cam_pose_cw) {
            const bool recovered = previous_lost_;
            tracking_state = recovered ? "recovering" : "tracking";
            reason = recovered ? "stella_tracking_recovered" : "stella_pose_valid";
            const Eigen::Matrix4d T_camera_from_reconstruction = *cam_pose_cw;
            const Eigen::Matrix4d T_reconstruction_from_camera = T_camera_from_reconstruction.inverse();
            pose = {
                {"pose_id", "stella-live-pose-" + std::to_string(sequence_index)},
                {"camera_id", camera_id_},
                {"frame_id", frame_id},
                {"timestamp_ns", timestamp_ns},
                {"camera_frame_id", camera_frame_id_},
                {"reconstruction_frame_id", kReconstructionFrame},
                {"pose_convention", kPoseOutputConvention},
                {"T_reconstruction_from_camera", matrix_to_json(T_reconstruction_from_camera)},
                {"quaternion_order", "wxyz"},
                {"gauge", {
                    {"scale_status", "arbitrary"},
                    {"alignment_status", "arbitrary"},
                    {"unit_to_metre", nullptr},
                    {"scale_source", ""},
                    {"relative_scale_uncertainty", nullptr},
                }},
            };
            if (recovered) {
                ++relocalization_count_;
                last_relocalization_frame_id_ = frame_id;
            }
            ever_had_pose_ = true;
            previous_lost_ = false;
        } else if (ever_had_pose_) {
            tracking_state = "lost";
            reason = "stella_tracking_lost";
            previous_lost_ = true;
        } else {
            tracking_state = "initializing";
            reason = "stella_initializing";
        }

        std::string keyframe_id;
        if (is_keyframe) {
            unsigned int newest_id = 0;
            bool have_id = false;
            for (const auto& keyframe : keyframes) {
                if (!keyframe) continue;
                if (!have_id || keyframe->id_ > newest_id) {
                    newest_id = keyframe->id_;
                    have_id = true;
                }
            }
            if (have_id) keyframe_id = "stella-keyframe-" + std::to_string(newest_id);
        }
        json map_snapshot = {
            {"revision", map_revision_},
            {"keyframe_count", keyframe_count},
            {"landmark_count", landmark_count},
        };
        const bool publish_sparse_map =
            map_revision_ != last_sparse_map_revision_ ||
            sequence_index % kSparseMapPublishIntervalFrames == 0;
        if (publish_sparse_map) {
            json samples = json::array();
            const std::size_t total = landmarks.size();
            const std::size_t stride = total > kMaxSparseLandmarkSamples
                ? (total + kMaxSparseLandmarkSamples - 1) / kMaxSparseLandmarkSamples
                : 1;
            for (std::size_t index = 0; index < total && samples.size() < kMaxSparseLandmarkSamples; index += stride) {
                const auto& landmark = landmarks[index];
                if (!landmark) continue;
                const auto pos = landmark->get_pos_in_world();
                if (!std::isfinite(pos(0)) || !std::isfinite(pos(1)) || !std::isfinite(pos(2))) continue;
                samples.push_back({pos(0), pos(1), pos(2)});
            }
            map_snapshot["landmarks_reconstruction"] = std::move(samples);

            // Publish a bounded, deterministic sample of optimized historical
            // keyframe poses alongside the sparse landmarks.  These poses stay
            // in the same arbitrary-gauge reconstruction frame and are used
            // only for C4 map-structure presentation.
            std::vector<std::shared_ptr<stella_vslam::data::keyframe>> ordered_keyframes;
            ordered_keyframes.reserve(keyframes.size());
            for (const auto& keyframe : keyframes) {
                if (keyframe) ordered_keyframes.push_back(keyframe);
            }
            std::sort(
                ordered_keyframes.begin(), ordered_keyframes.end(),
                [](const auto& lhs, const auto& rhs) { return lhs->id_ < rhs->id_; });
            json keyframe_samples = json::array();
            const std::size_t keyframe_total = ordered_keyframes.size();
            const std::size_t keyframe_emit =
                std::min(keyframe_total, kMaxLiveKeyframePoseSamples);
            for (std::size_t sample_index = 0; sample_index < keyframe_emit; ++sample_index) {
                const std::size_t source_index = keyframe_emit <= 1
                    ? 0
                    : (sample_index * (keyframe_total - 1)) / (keyframe_emit - 1);
                const auto& keyframe = ordered_keyframes[source_index];
                const Eigen::Matrix4d T_reconstruction_from_camera = keyframe->get_pose_wc();
                if (!T_reconstruction_from_camera.allFinite()) continue;
                const std::uint64_t keyframe_timestamp_ns = static_cast<std::uint64_t>(
                    std::max(0.0, keyframe->timestamp_) * 1.0e9 + 0.5);
                keyframe_samples.push_back({
                    {"keyframe_id", "stella-keyframe-" + std::to_string(keyframe->id_)},
                    {"timestamp_ns", keyframe_timestamp_ns},
                    {"pose_convention", kPoseOutputConvention},
                    {"T_reconstruction_from_camera", matrix_to_json(T_reconstruction_from_camera)},
                });
            }
            map_snapshot["keyframes_reconstruction"] = std::move(keyframe_samples);
            last_sparse_map_revision_ = map_revision_;
        }

        json observation = {
            {"session_id", options_.session_id},
            {"frame_id", frame_id},
            {"sequence_index", sequence_index},
            {"tracking_state", tracking_state},
            {"quality", {
                {"tracked_features", 0},
                {"inlier_features", 0},
                {"inlier_ratio", nullptr},
                {"tracking_confidence", nullptr},
                {"processing_latency_ns", latency_ns < 0 ? 0 : latency_ns},
                {"reason", reason},
            }},
            {"pose", pose},
            {"is_keyframe", is_keyframe},
            {"keyframe_id", keyframe_id},
            {"map_snapshot", map_snapshot},
        };
        channel_.send_message("observation", {{"observation", observation}});
    }

    void request_loop_closure(const json& payload) {
        if (!running_ || !slam_) fail("Loop closure request requires an active stella session");
        if (!enable_loop_closure_ || !enable_global_optimization_) {
            fail("Loop closure/global optimization are disabled for this session");
        }
        const std::string from_id = payload.at("from_keyframe_id").get<std::string>();
        const std::string to_id = payload.at("to_keyframe_id").get<std::string>();
        const int from_native = parse_keyframe_id(from_id);
        const int to_native = parse_keyframe_id(to_id);
        if (from_native == to_native) fail("Loop closure requires two different keyframes");
        const bool accepted = slam_->request_loop_closure(from_native, to_native);
        if (accepted) ++accepted_loop_closure_requests_;
        channel_.send_message(
            "loop_closure_requested",
            {
                {"from_keyframe_id", from_id},
                {"to_keyframe_id", to_id},
                {"accepted", accepted},
                {"accepted_request_count", accepted_loop_closure_requests_},
            });
    }

    void send_global_consistency_status() {
        if (!running_ || !slam_ || !map_publisher_) {
            fail("Global-consistency status requires an active stella session");
        }
        std::vector<std::shared_ptr<stella_vslam::data::keyframe>> keyframes;
        map_publisher_->get_keyframes(keyframes);
        refresh_optimized_pose_revision(keyframes);
        std::sort(
            keyframes.begin(), keyframes.end(),
            [](const auto& lhs, const auto& rhs) {
                if (!lhs) return false;
                if (!rhs) return true;
                return lhs->id_ < rhs->id_;
            });
        json poses = json::array();
        std::size_t emitted_keyframes = 0;
        for (const auto& keyframe : keyframes) {
            if (!keyframe) continue;
            const Eigen::Matrix4d T_reconstruction_from_camera = keyframe->get_pose_wc();
            const std::string keyframe_id = "stella-keyframe-" + std::to_string(keyframe->id_);
            const auto timestamp_ns = static_cast<std::uint64_t>(
                std::max(0.0, keyframe->timestamp_) * 1.0e9 + 0.5);
            poses.push_back({
                {"keyframe_id", keyframe_id},
                {"frame_id", keyframe_id},
                {"timestamp_ns", timestamp_ns},
                {"source_pose_id", "stella-native-keyframe-pose-" + std::to_string(keyframe->id_)},
                {"optimization_revision", optimized_pose_revision_},
                {"pose_layer", "optimized"},
                {"pose_convention", kPoseOutputConvention},
                {"T_reconstruction_from_camera", matrix_to_json(T_reconstruction_from_camera)},
            });
            ++emitted_keyframes;
        }
        std::vector<std::shared_ptr<stella_vslam::data::landmark>> landmarks;
        std::set<std::shared_ptr<stella_vslam::data::landmark>> local_landmarks;
        const unsigned int landmark_count = map_publisher_->get_landmarks(landmarks, local_landmarks);
        channel_.send_message(
            "global_consistency_status",
            {
                {"loop_detector_enabled", slam_->loop_detector_is_enabled()},
                {"global_optimization_enabled", enable_global_optimization_},
                {"loop_ba_running", slam_->loop_BA_is_running()},
                {"map_revision", map_revision_},
                {"optimized_pose_revision", optimized_pose_revision_},
                {"keyframe_count", emitted_keyframes},
                {"landmark_count", landmark_count},
                {"accepted_loop_closure_requests", accepted_loop_closure_requests_},
                {"relocalization_count", relocalization_count_},
                {"last_relocalization_frame_id", last_relocalization_frame_id_},
                {"optimized_keyframe_poses", poses},
                {"pose_convention", kPoseOutputConvention},
                {"scale_status", "arbitrary"},
                {"unit_to_metre", nullptr},
            });
    }

    void stop() {
        if (!running_) return;
        if (slam_) slam_->shutdown();
        slam_.reset();
        map_publisher_.reset();
        config_.reset();
        ring_.close();
        previous_optimized_poses_.clear();
        running_ = false;
    }

    bool running() const { return running_; }

private:
    static int parse_keyframe_id(const std::string& value) {
        static const std::string prefix = "stella-keyframe-";
        if (value.rfind(prefix, 0) != 0 || value.size() <= prefix.size()) {
            fail("Native stella keyframe ID must use stella-keyframe-N format");
        }
        std::size_t consumed = 0;
        long long parsed = 0;
        try {
            parsed = std::stoll(value.substr(prefix.size()), &consumed, 10);
        } catch (const std::exception&) {
            fail("Native stella keyframe ID contains an invalid integer");
        }
        if (consumed != value.size() - prefix.size() || parsed < 0 || parsed > 2147483647LL) {
            fail("Native stella keyframe ID contains an out-of-range integer");
        }
        return static_cast<int>(parsed);
    }

    void refresh_optimized_pose_revision(
        const std::vector<std::shared_ptr<stella_vslam::data::keyframe>>& keyframes) {
        bool existing_pose_changed = false;
        std::map<unsigned int, Eigen::Matrix4d> current;
        for (const auto& keyframe : keyframes) {
            if (!keyframe) continue;
            const Eigen::Matrix4d pose = keyframe->get_pose_wc();
            current.emplace(keyframe->id_, pose);
            const auto previous = previous_optimized_poses_.find(keyframe->id_);
            if (previous != previous_optimized_poses_.end()) {
                const double difference = (previous->second - pose).norm();
                if (std::isfinite(difference) && difference > 1.0e-9) {
                    existing_pose_changed = true;
                }
            }
        }
        if (!previous_optimized_poses_.empty() && existing_pose_changed) {
            ++optimized_pose_revision_;
        }
        previous_optimized_poses_ = std::move(current);
    }

    void stop_noexcept() noexcept {
        try { stop(); } catch (...) {}
    }

    const Options& options_;
    JsonLineChannel& channel_;
    RingSpec ring_spec_;
    ReadOnlyRing ring_;
    std::shared_ptr<stella_vslam::config> config_;
    std::shared_ptr<stella_vslam::system> slam_;
    std::shared_ptr<stella_vslam::publish::map_publisher> map_publisher_;
    std::string camera_id_;
    std::string camera_frame_id_;
    int width_ = 0;
    int height_ = 0;
    bool enable_mapping_ = true;
    bool enable_loop_closure_ = false;
    bool enable_global_optimization_ = false;
    bool running_ = false;
    bool ever_had_pose_ = false;
    bool previous_lost_ = false;
    unsigned int previous_keyframes_ = 0;
    unsigned int previous_landmarks_ = 0;
    std::uint64_t map_revision_ = 0;
    std::uint64_t last_sparse_map_revision_ = static_cast<std::uint64_t>(-1);
    std::uint64_t optimized_pose_revision_ = 1;
    std::uint64_t accepted_loop_closure_requests_ = 0;
    std::uint64_t relocalization_count_ = 0;
    std::string last_relocalization_frame_id_;
    std::map<unsigned int, Eigen::Matrix4d> previous_optimized_poses_;
};

void send_error(JsonLineChannel& channel, const std::string& code, const std::string& message, bool retryable = false) {
    channel.send_message(
        "error",
        {{"code", code}, {"message", message}, {"retryable", retryable}});
}

int run_worker(const Options& options) {
    UnixServer server(options.socket_path);
    server.accept_one();
    JsonLineChannel channel(server.client_fd(), options.session_id);
    channel.send_message(
        "hello",
        {
            {"backend_id", options.backend_id},
            {"worker_version", kWorkerVersion},
            {"stella_version", EIA_STELLA_VERSION},
            {"stella_commit", EIA_STELLA_COMMIT},
            {"pose_input_convention", kPoseInputConvention},
            {"pose_output_convention", kPoseOutputConvention},
            {"pixel_transport", "posix_shared_memory"},
            {"control_transport", "af_unix_json_lines"},
            {"sparse_landmark_samples_supported", true},
            {"sparse_landmark_sample_cap", kMaxSparseLandmarkSamples},
        });
    channel.send_message(
        "ready",
        {
            {"protocol_version", kProtocolVersion},
            {"loop_closure_supported_in_phase8_4", false},
            {"loop_closure_supported_in_phase8_5", true},
            {"global_optimization_supported_in_phase8_5", true},
            {"sparse_landmark_samples_supported", true},
            {"sparse_landmark_sample_cap", kMaxSparseLandmarkSamples},
            {"default_loop_closure_enabled", false},
            {"default_global_optimization_enabled", false},
            {"metric_scale", false},
        });

    StellaSession session(options, channel);
    for (;;) {
        json message = channel.receive_message();
        const std::string type = message.at("message_type").get<std::string>();
        const json& payload = message.at("payload");
        try {
            if (type == "start") {
                session.start(payload);
            } else if (type == "frame_available") {
                session.process_frame(payload);
            } else if (type == "request_loop_closure") {
                session.request_loop_closure(payload);
            } else if (type == "global_consistency_status_request") {
                session.send_global_consistency_status();
            } else if (type == "stop") {
                session.stop();
                channel.send_message("stopped", {{"clean", true}});
            } else if (type == "heartbeat") {
                channel.send_message("heartbeat", {{"alive", true}, {"running", session.running()}});
            } else if (type == "shutdown") {
                session.stop();
                return 0;
            } else {
                send_error(channel, "unexpected_message", "Unsupported message_type: " + type);
            }
        } catch (const std::exception& exc) {
            send_error(channel, "worker_operation_failed", exc.what(), false);
            if (type == "start") session.stop();
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        return run_worker(options);
    } catch (const std::exception& exc) {
        std::cerr << "EIA live SLAM worker fatal: " << exc.what() << std::endl;
        return 2;
    }
}
