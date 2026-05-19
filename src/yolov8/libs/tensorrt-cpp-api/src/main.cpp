// Standalone benchmark for the tensorrt_cpp_api library. Not used by the ROS node.
// Loads an ONNX model, builds (or reuses) a TensorRT engine, and times warmup + inference.
//
// Usage:
//   run_inference_benchmark /path/to/model.onnx /path/to/image.jpg
#include <chrono>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "engine.h"

namespace {
constexpr size_t kWarmupIterations = 100;
constexpr size_t kBenchmarkIterations = 1000;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " /path/to/onnx/model.onnx /path/to/image.jpg" << std::endl;
        return 1;
    }

    const std::string onnxModelPath = argv[1];
    const std::string inputImagePath = argv[2];

    if (!Util::doesFileExist(onnxModelPath)) {
        std::cerr << "Error: model not found at " << onnxModelPath << std::endl;
        return 1;
    }
    if (!Util::doesFileExist(inputImagePath)) {
        std::cerr << "Error: image not found at " << inputImagePath << std::endl;
        return 1;
    }

    // Single-image benchmark: batch size 1.
    Options options;
    options.precision = Precision::FP16;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;

    Engine engine(options);

    // YOLOv8 expects inputs in [0, 1] with no mean subtraction.
    const std::array<float, 3> subVals{0.f, 0.f, 0.f};
    const std::array<float, 3> divVals{1.f, 1.f, 1.f};
    const bool normalize = true;

    if (!engine.build(onnxModelPath, subVals, divVals, normalize)) {
        std::cerr << "Error: unable to build TRT engine." << std::endl;
        return 1;
    }
    if (!engine.loadNetwork(onnxModelPath)) {
        std::cerr << "Error: unable to load TRT engine." << std::endl;
        return 1;
    }

    auto cpuImg = cv::imread(inputImagePath);
    if (cpuImg.empty()) {
        std::cerr << "Error: unable to read image at " << inputImagePath << std::endl;
        return 1;
    }

    cv::cuda::GpuMat img;
    img.upload(cpuImg);
    cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

    const auto& inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    const size_t batchSize = options.optBatchSize;
    for (const auto & inputDim : inputDims) {
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j) {
            auto resized = Engine::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    std::cout << "Warming up (" << kWarmupIterations << " iterations)..." << std::endl;
    std::vector<std::vector<std::vector<float>>> featureVectors;
    for (size_t i = 0; i < kWarmupIterations; ++i) {
        if (!engine.runInference(inputs, featureVectors)) {
            std::cerr << "Error: warmup inference failed." << std::endl;
            return 1;
        }
    }

    std::cout << "Benchmarking (" << kBenchmarkIterations << " iterations)..." << std::endl;
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < kBenchmarkIterations; ++i) {
        featureVectors.clear();
        if (!engine.runInference(inputs, featureVectors)) {
            std::cerr << "Error: benchmark inference failed." << std::endl;
            return 1;
        }
    }
    const auto totalElapsedMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    const auto avgPerSampleMs = totalElapsedMs / kBenchmarkIterations / static_cast<float>(inputs[0].size());

    std::cout << "\n=== Benchmark complete ===" << std::endl;
    std::cout << "Batch size:     " << inputs[0].size() << std::endl;
    std::cout << "Avg/sample:     " << avgPerSampleMs << " ms" << std::endl;
    std::cout << "Avg throughput: " << static_cast<int>(1000 / avgPerSampleMs) << " fps" << std::endl;
    std::cout << "==========================\n" << std::endl;

    // Print a preview of each output feature vector.
    for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
        for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
            std::cout << "Batch " << batch << ", output " << outputNum << ": ";
            int printed = 0;
            for (const auto &value : featureVectors[batch][outputNum]) {
                std::cout << value << " ";
                if (++printed == 10) {
                    std::cout << "...";
                    break;
                }
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
