#pragma once

// Vendored from https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP (MIT, see ./LICENSE).
// This file is intentionally kept close to upstream so future syncs are tractable. Bug fixes
// and small cleanups are applied; structural refactors are not. Logging here uses std::cout/cerr
// because the library is designed to be usable outside ROS.

#include <fstream>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
// Suppress -Wunused-parameter inside the TensorRT public headers (out of our control).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "NvInfer.h"
#pragma GCC diagnostic pop

namespace Util {
    inline bool doesFileExist(const std::string& filepath) {
        std::ifstream f(filepath.c_str());
        return f.good();
    }

    inline void checkCudaErrorCode(cudaError_t code) {
        if (code != cudaSuccess) {
            const std::string errMsg = "CUDA operation failed with code "
                + std::to_string(code) + " (" + cudaGetErrorName(code)
                + "): " + cudaGetErrorString(code);
            std::cerr << errMsg << std::endl;
            throw std::runtime_error(errMsg);
        }
    }

    std::vector<std::string> getFilesInDirectory(const std::string& dirPath);
}

// Monotonic stopwatch. Only the standalone benchmark executable (src/main.cpp) uses it; it is
// kept in the public header because the library is published for general (non-ROS) use.
template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch {
    typename Clock::time_point start_point;
public:
    Stopwatch() : start_point(Clock::now()) {}

    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsedTime() const {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};

using preciseStopwatch = Stopwatch<>;

// Precision used for GPU inference
enum class Precision {
    // Full precision floating point value
    FP32,
    // Half precision floating point value
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
};

// Options for the network. The defaults below are the upstream library defaults; the YoloV8
// wrapper in this project sets both optBatchSize and maxBatchSize from YoloV8Config::batchSize,
// so the mismatched defaults here are not used by the ROS node.
struct Options {
    Precision precision = Precision::FP16;
    // If INT8 precision is selected, must provide path to calibration dataset directory.
    std::string calibrationDataDirectoryPath;
    // Batch size used while computing INT8 calibration data; set as large as the GPU allows.
    int32_t calibrationBatchSize = 128;
    // Build-time scratch budget handed to TensorRT for tactic selection.
    size_t maxWorkspaceSize = 4ULL << 30;
    int32_t optBatchSize = 4;
    int32_t maxBatchSize = 6;
    int deviceIndex = 0;
};

// Class used for int8 calibration
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH, const std::string& calibDataDirPath, const std::string& calibTableName, const std::string& inputBlobName,
                           const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},const std::array<float, 3>& divVals = {1.f, 1.f, 1.f}, bool normalize = true, bool readCache = true);
    virtual ~Int8EntropyCalibrator2();
    // Abstract base class methods which must be implemented
    int32_t getBatchSize () const noexcept override;
    bool getBatch (void *bindings[], char const *names[], int32_t nbBindings) noexcept override;
    void const * readCalibrationCache (std::size_t &length) noexcept override;
    void writeCalibrationCache (void const *ptr, std::size_t length) noexcept override;
private:
    const int32_t m_batchSize;
    const int32_t m_inputW;
    const int32_t m_inputH;
    int32_t m_imgIdx;
    std::vector<std::string> m_imgPaths;
    size_t m_inputCount;
    const std::string m_calibTableName;
    const std::string m_inputBlobName;
    const std::array<float, 3> m_subVals;
    const std::array<float, 3> m_divVals;
    const bool m_normalize;
    const bool m_readCache;
    void* m_deviceInput;
    std::vector<char> m_calibCache;
};

/**
 * The logger TensorRT needs for the Builder, ONNX Parser, and Runtime interfaces.
 */
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class Engine {
public:
    Engine(const Options& options);
    ~Engine();
    // Build the network
    // The default implementation will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f] (some converted models may require this).
    // If the model requires values to be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    bool build(std::string onnxModelPath, const std::array<float, 3>& subVals = {0.f, 0.f, 0.f}, const std::array<float, 3>& divVals = {1.f, 1.f, 1.f},
               bool normalize = true);
    // Load and prepare the network for inference
    bool loadNetwork(std::string onnxModelPath);
    // Run inference.
    // Input format [input][batch][cv::cuda::GpuMat]
    // Output format [batch][output][feature_vector]
    // `downloadOutput` selects, per output tensor, whether it is copied back to the host. An
    // empty vector downloads everything (the original behaviour). Outputs left on the device
    // are reachable via getOutputDevicePtr() and stay valid until the next runInference call.
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>>& inputs,
                      std::vector<std::vector<std::vector<float>>>& featureVectors,
                      const std::vector<bool>& downloadOutput = {});

    // Device pointer to output tensor `outputIdx` for batch item `batch`.
    [[nodiscard]] void* getOutputDevicePtr(int outputIdx, int batch) const {
        const size_t numInputs = m_inputDims.size();
        const size_t len = m_outputLengthsFloat[outputIdx];
        return static_cast<char*>(m_buffers[numInputs + outputIdx]) +
               static_cast<size_t>(batch) * len * sizeof(float);
    }

    // Utility method for resizing an image while maintaining the aspect ratio by adding padding to smaller dimension after scaling
    // While letterbox padding normally adds padding to top & bottom, or left & right sides, this implementation only adds padding to the right or bottom side
    // This is done so that it's easier to convert detected coordinates (ex. YOLO model) back to the original reference frame.
    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat& input, size_t height, size_t width, const cv::Scalar& bgcolor = cv::Scalar(0, 0, 0));

    [[nodiscard]] const std::vector<nvinfer1::Dims3>& getInputDims() const { return m_inputDims; }
    [[nodiscard]] const std::vector<nvinfer1::Dims>& getOutputDims() const { return m_outputDims; }
    [[nodiscard]] int getBatchSize() const { return batch_size_; }

    // Converts a batch of NHWC GpuMats into a single NCHW GpuMat blob, applying optional
    // normalization and mean subtraction in one pass.
    static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batchInput, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals, bool normalize);
    // Same conversion but into engine-owned scratch and on the engine's stream. Used on the
    // hot path; the static version stays for the offline INT8 calibrator.
    cv::cuda::GpuMat blobFromGpuMatsCached(const std::vector<cv::cuda::GpuMat>& batchInput, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals, bool normalize);
private:
    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options& options, const std::string& onnxModelPath);

    void getDeviceNames(std::vector<std::string>& deviceNames);

    int batch_size_;

    // Normalization, scaling, and mean subtraction of inputs
    std::array<float, 3> m_subVals{};
    std::array<float, 3> m_divVals{};
    bool m_normalize;

    // Holds pointers to the input and output GPU buffers
    std::vector<void*> m_buffers;
    std::vector<uint32_t> m_outputLengthsFloat{};
    // Created once in loadNetwork. Creating/destroying a stream per inference costs a
    // device sync on every frame.
    cudaStream_t m_inferenceStream = nullptr;
    // Waiting on a blocking-sync event parks the thread on a driver semaphore. Plain
    // cudaStreamSynchronize spins, which burns a core for the whole inference.
    cudaEvent_t m_inferenceDone = nullptr;
    // Scratch for the NCHW blob, reused so the hot path does no device allocation.
    cv::cuda::GpuMat m_blob;
    // Preprocess scratch, allocated once. Every cudaMalloc/cudaFree serialises with the
    // device, and the old code did ~14 of them per inference.
    std::vector<cv::cuda::GpuMat> m_letterbox;
    std::vector<cv::cuda::GpuMat> m_letterboxTmp;
    cv::cuda::GpuMat m_blobFloat;
    cv::cuda::Stream m_cvStream;

public:
    // Letterbox into reusable scratch owned by the engine; `slot` indexes the batch.
    cv::cuda::GpuMat letterboxInto(const cv::cuda::GpuMat& input, size_t slot,
                                   size_t height, size_t width);
private:
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;

    // Must keep IRuntime around for inference, see: https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<Int8EntropyCalibrator2> m_calibrator = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options m_options;
    Logger m_logger;
    std::string m_engineName;
};
