#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <iterator>
#include <opencv2/cudaimgproc.hpp>
#include "engine.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;
using namespace Util;

std::vector<std::string> Util::getFilesInDirectory(const std::string& dirPath) {
    std::vector<std::string> filepaths;
    for (const auto& entry: std::filesystem::directory_iterator(dirPath)) {
        filepaths.emplace_back(entry.path());
    }
    return filepaths;
}

/**
 * Creates the logs that TensorRT needs for the Builder, ONNX Parser, and Runtime interfaces.
 *
 * @param severity The severity level of the log message.
 * @param msg The message to log.
 */
void Logger::log(Severity severity, const char *msg) noexcept {
    // Only log warnings or more severe (kINTERNAL_ERROR=0, kERROR=1, kWARNING=2).
    // Increase to kINFO/kVERBOSE when debugging engine builds.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

Engine::Engine(const Options &options)
    : m_options(options) {}

/**
 * Builds the TensorRT engine by converted the specified ONNX model
 * using the ONNX Parser and creating a Builder to build the engine
 * with engine options.
 * 
 * @param onnxModelPath The path to the ONNX model file.
 * @param subVals The array of subtraction values used for input preprocessing.
 * @param divVals The array of division values used for input preprocessing.
 * @param normalize Flag indicating whether input normalization should be applied.
 * @return True if the engine is successfully built and saved, false otherwise.
 * @throws std::runtime_error if the model file is not found or if there are errors during engine generation.
 */
bool Engine::build(std::string onnxModelPath, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals,
                   bool normalize) {
    m_subVals = subVals;
    m_divVals = divVals;
    m_normalize = normalize;

    // Get models directory from onnxModelPath
    std::string modelsDir = onnxModelPath.substr(0, onnxModelPath.find_last_of("/"));

    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options, onnxModelPath);
    batch_size_ = m_options.maxBatchSize;
    std::cout << "Searching for engine file " << m_engineName << " in directory " << modelsDir << std::endl;

    // Create Engine path
    std::string engine_path = modelsDir + "/engines/";
    if (!std::filesystem::exists(engine_path)) {
        std::filesystem::create_directory(engine_path);
    }
    engine_path += m_engineName;

    if (doesFileExist(engine_path)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    if (!doesFileExist(onnxModelPath)) {
        throw std::runtime_error("Could not find model at path: " + onnxModelPath);
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating. This could take a while..." << std::endl;

    // Create the TensorRT Build
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Set kEXPLICIT_BATCH flag via bit shifting for the NetworkDefinition as the ONNX PARSER does not support implicit batch sizes
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // Create the NetworkDefinition for the Build and specify the explicit batch flag
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        throw std::runtime_error("Error, model needs at least 1 input!");
    }

    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            throw std::runtime_error("Error, the model has multiple inputs, each with differing batch sizes!");
        }
    }

    // Ensure the imported ONNX model supports the max batch size specified or supports dynamic batching
    if (m_options.maxBatchSize > input0Batch && input0Batch != -1) {
        throw std::runtime_error("Error, imported ONNX model does not support max batch size of " +
                                std::to_string(m_options.maxBatchSize) + ". The ONNX model only supports a max batch size of " +
                                std::to_string(input0Batch) + ".");
    }

    // Create a builder configuration
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // We pin min/opt/max to maxBatchSize so the engine is specialized to exactly that batch size.
        // The model still must be exported with that batch (or dynamic batch) for this to succeed.
        optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);

    // Set the precision level
    std::string precisionLabel = "FP32";
    if (m_options.precision == Precision::FP16) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("Error: GPU does not support FP16 precision");
        }
        config->setFlag(BuilderFlag::kFP16);
        precisionLabel = "FP16";
    } else if (m_options.precision == Precision::INT8) {
        if (numInputs > 1) {
            throw std::runtime_error("Error, this implementation currently only supports INT8 quantization for single input models");
        }

        // Ensure the GPU supports INT8 Quantization
        if (!builder->platformHasFastInt8()) {
            throw std::runtime_error("Error: GPU does not support INT8 precision");
        }

        // Ensure the user has provided path to calibration data directory
        if (m_options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: If INT8 precision is selected, must provide path to calibration data directory to Engine::build method");
        }

        config->setFlag((BuilderFlag::kINT8));

        const auto input = network->getInput(0);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        const auto calibrationFileName = m_engineName + ".calibration";

        m_calibrator = std::make_unique<Int8EntropyCalibrator2>(m_options.calibrationBatchSize, inputDims.d[3], inputDims.d[2], m_options.calibrationDataDirectoryPath,
                                                                calibrationFileName, inputName, subVals, divVals, normalize);
        config->setInt8Calibrator(m_calibrator.get());
        precisionLabel = "INT8";
    }
    std::cout << "Precision set to " << precisionLabel << std::endl;

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine. If this call fails, raise the logger to kVERBOSE for diagnostics.
    std::cout << "Starting engine build..." << std::endl;
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        std::cout << "Error, engine build failed!" << std::endl;
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(engine_path, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

Engine::~Engine() {
    // Free the GPU memory. Log instead of throwing so a failure during teardown doesn't trigger
    // std::terminate from a destructor.
    for (auto & buffer : m_buffers) {
        const cudaError_t code = cudaFree(buffer);
        if (code != cudaSuccess) {
            std::cerr << "Engine::~Engine cudaFree failed: " << cudaGetErrorString(code) << std::endl;
        }
    }
    m_buffers.clear();
}

/**
 * @brief Deserializes engine from disk, creates a TensorRT ExecutionContext, and allocates tensors
 *      on the GPU for input and output buffers.
 *
 * @param onnxModelPath The path to the ONNX model file directory.
 * @return true if the network is loaded successfully, false otherwise.
 * @throws std::runtime_error if there is an error reading the engine file or setting the GPU device index.
 */
bool Engine::loadNetwork(std::string onnxModelPath) {
    // Get engine path from onnxModelPath
    std::string modelsDir = onnxModelPath.substr(0, onnxModelPath.find_last_of("/"));
    std::string engine_path = modelsDir + "/engines/" + m_engineName;

    // Read the serialized model from disk
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<IRuntime> {createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // The execution context contains all of the state associated with a particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    m_buffers.resize(m_engine->getNbIOTensors());

    // Create a cuda stream
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    m_outputLengthsFloat.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        if (tensorType == TensorIOMode::kINPUT) {
            // Allocate memory for the input
            // Allocate enough to fit the max batch size (we could end up using less later)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], m_options.maxBatchSize * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float), stream));

            // Store the input dims for later use
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
        } else if (tensorType == TensorIOMode::kOUTPUT) {
            // The binding is an output
            uint32_t outputLenFloat = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= tensorShape.d[j];
            }

            m_outputLengthsFloat.push_back(outputLenFloat);
            // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_options.maxBatchSize * sizeof(float), stream));
        } else {
            throw std::runtime_error("Error, IO Tensor is neither an input or output!");
        }
    }

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

bool Engine::runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs, std::vector<std::vector<std::vector<float>>>& featureVectors) {
    if (inputs.empty() || inputs[0].empty()) {
        std::cerr << "Engine::runInference error: provided input vector is empty" << std::endl;
        return false;
    }

    const auto numInputs = m_inputDims.size();

    // Ensure the batch size does not exceed the max
    if (inputs.size() > static_cast<size_t>(m_options.maxBatchSize)) {
        std::cerr << "Engine::runInference error: batch size " << inputs.size()
                  << " exceeds engine max batch size " << m_options.maxBatchSize << std::endl;
        return false;
    }

    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].size() != inputs[0].size()) {
            std::cerr << "Engine::runInference error: batch size must be constant for all inputs" << std::endl;
            return false;
        }
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Iterate through the inputs (there should only be one input)
    for (size_t i = 0; i < numInputs; ++i) {
        std::vector<cv::cuda::GpuMat> input_batches = inputs[i];
        const auto batchSize = static_cast<int32_t>(input_batches.size());
        const auto& dims = m_inputDims[i];

        // Check the dimensions of the first batched image match the engine dims.
        // Per-image validation is the caller's responsibility.
        const auto& batch0 = input_batches[0];
        if (batch0.channels() != dims.d[0] ||
            batch0.rows != dims.d[1] ||
            batch0.cols != dims.d[2]) {
            std::cerr << "Engine::runInference error: input has wrong shape. Expected ("
                      << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ") but got ("
                      << batch0.channels() << ", " << batch0.rows << ", " << batch0.cols << ")"
                      << std::endl;
            return false;
        }

        nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
        m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims);

        // OpenCV stores images NHWC, TensorRT expects NCHW. blobFromGpuMats converts and
        // applies normalization / mean subtraction in one pass.
        cv::cuda::GpuMat mfloat = blobFromGpuMats(input_batches, m_subVals, m_divVals, m_normalize);
        const auto blob_size = mfloat.cols * mfloat.rows * sizeof(float);
        checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], mfloat.ptr<void>(), blob_size,
                                           cudaMemcpyDeviceToDevice, inferenceCudaStream));
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        if (!m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i])) {
            std::cerr << "Engine::runInference error: setTensorAddress failed for "
                      << m_IOTensorNames[i] << std::endl;
            return false;
        }
    }

    if (!m_context->enqueueV3(inferenceCudaStream)) {
        std::cerr << "Engine::runInference error: enqueueV3 failed" << std::endl;
        return false;
    }

    // Copy outputs from GPU back to host. Output shape is [batch][output][feature_vector].
    featureVectors.clear();
    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    const auto numTensors = static_cast<int32_t>(m_buffers.size());
    for (int batch = 0; batch < batchSize; ++batch) {
        std::vector<std::vector<float>> outputs{};
        // Outputs follow inputs in m_buffers, indexed by the engine's IO tensor list.
        for (int32_t outputBinding = static_cast<int32_t>(numInputs); outputBinding < numTensors; ++outputBinding) {
            const auto outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
            std::vector<float> output(outputLenFloat);
            checkCudaErrorCode(cudaMemcpyAsync(output.data(),
                                               static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat),
                                               outputLenFloat * sizeof(float),
                                               cudaMemcpyDeviceToHost, inferenceCudaStream));
            outputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(outputs));
    }

    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}


/**
 * Converts a batch of GPU mats (a batch of tensors) to a single GPU contiguous mat blob in GPU memory (one tensor).
 *
 * @param batches The vector of GPU mats representing the input batches.
 * @param subVals The array of three float values used for mean subtraction.
 * @param divVals The array of three float values used for scaling.
 * @param normalize A boolean flag indicating whether to normalize the output.
 * @return The GPU mat blob representing the converted batch.
 */
cv::cuda::GpuMat Engine::blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batches, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals, bool normalize) {
    const size_t batch_size = batches.size();
    const int channels = batches[0].channels();
    const int height = batches[0].rows;
    const int width = batches[0].cols;

    const size_t img_size = static_cast<size_t>(height) * width * channels;
    cv::cuda::GpuMat gpu_dst(static_cast<int>(batch_size), static_cast<int>(img_size), CV_8UC3);

    const size_t img_channel_size = static_cast<size_t>(width) * height;
    for (size_t batch = 0; batch < batch_size; batch++) {
        std::vector<cv::cuda::GpuMat> input_channels(channels);
        for (int j = 0; j < channels; j++) {
            input_channels[j] = cv::cuda::GpuMat(height, width, CV_8UC1,
                &(gpu_dst.ptr()[img_size * batch + img_channel_size * j]));
        }
        cv::cuda::split(batches[batch], input_channels);  // HWC -> CHW
    }

    // Normalize the images
    cv::cuda::GpuMat mfloat;
    if (normalize) {
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    } else {
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }

    cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return mfloat;
}

std::string Engine::serializeEngineOptions(const Options &options, const std::string& onnxModelPath) {
    const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
    std::string engineName = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos);

    // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    auto deviceName = deviceNames[options.deviceIndex];
    // Remove spaces from the device name
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

    engineName+= "." + deviceName;

    // Serialize the specified options into the filename
    if (options.precision == Precision::FP16) {
        engineName += "fp16";
    } else if (options.precision == Precision::FP32){
        engineName += "fp32";
    } else {
        engineName += "int8";
    }

    engineName += "maxBatchSize" + std::to_string(options.maxBatchSize);
    engineName += "optimalBatchSize" + std::to_string(options.optBatchSize);
    engineName += ".engine";

    return engineName;
}

void Engine::getDeviceNames(std::vector<std::string>& deviceNames) {
    int numGPUs = 0;
    cudaError_t ret = cudaGetDeviceCount(&numGPUs);
    if (ret != cudaSuccess) {
        throw std::runtime_error("Error: could not determine number of CUDA-capable devices: "
                                 + std::string(cudaGetErrorString(ret)));
    }

    for (int device = 0; device < numGPUs; device++) {
        cudaDeviceProp prop;
        checkCudaErrorCode(cudaGetDeviceProperties(&prop, device));
        deviceNames.push_back(std::string(prop.name));
    }
}

/**
 * Resizes the input image while maintaining the aspect ratio and pads the right and bottom sides if necessary.
 *
 * @param input The input image to be resized.
 * @param height The desired height of the output image.
 * @param width The desired width of the output image.
 * @param bgcolor The background color to be used for padding.
 * @return The resized image with the specified dimensions and padding.
 */
cv::cuda::GpuMat Engine::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width, const cv::Scalar &bgcolor) {
    // Calculate a scaling factor to maintain the aspect ratio
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    // Resize the image to the new dimensions
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    // Create a new image with the desired dimensions and fill it with the background color
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    // Copy the resized image to the top left corner of the new image with padding and background color
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH,
                                               const std::string &calibDataDirPath,
                                               const std::string &calibTableName,
                                               const std::string &inputBlobName,
                                               const std::array<float, 3>& subVals,
                                               const std::array<float, 3>& divVals,
                                               bool normalize,
                                               bool readCache)
        : m_batchSize(batchSize)
        , m_inputW(inputW)
        , m_inputH(inputH)
        , m_imgIdx(0)
        , m_calibTableName(calibTableName)
        , m_inputBlobName(inputBlobName)
        , m_subVals(subVals)
        , m_divVals(divVals)
        , m_normalize(normalize)
        , m_readCache(readCache) {

    // Allocate GPU memory to hold the entire batch
    m_inputCount = 3 * inputW * inputH * batchSize;
    checkCudaErrorCode(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));

    // Read the name of all the files in the specified directory.
    if (!doesFileExist(calibDataDirPath)) {
        throw std::runtime_error("Error, directory at provided path does not exist: " + calibDataDirPath);
    }

    m_imgPaths = getFilesInDirectory(calibDataDirPath);
    if (m_imgPaths.size() < static_cast<size_t>(batchSize)) {
        throw std::runtime_error("There are fewer calibration images than the specified batch size!");
    }

    // Randomize the calibration data
    auto rd = std::random_device {};
    auto rng = std::default_random_engine { rd() };
    std::shuffle(std::begin(m_imgPaths), std::end(m_imgPaths), rng);
}

int32_t Int8EntropyCalibrator2::getBatchSize() const noexcept {
    // Return the batch size
    return m_batchSize;
}

bool Int8EntropyCalibrator2::getBatch(void **bindings, const char **names, int32_t nbBindings) noexcept {
    // This method will read a batch of images into GPU memory, and place the pointer to the GPU memory in the bindings variable.

    if (m_imgIdx + m_batchSize > static_cast<int>(m_imgPaths.size())) {
        // There are not enough images left to satisfy an entire batch
        return false;
    }

    // Read the calibration images into memory for the current batch
    std::vector<cv::cuda::GpuMat> inputImgs;
    for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++) {
        auto cpuImg = cv::imread(m_imgPaths[i]);
        if (cpuImg.empty()) {
            std::cerr << "Int8EntropyCalibrator2::getBatch error: unable to read image at "
                      << m_imgPaths[i] << std::endl;
            return false;
        }

        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(cpuImg);
        cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

        auto resized = Engine::resizeKeepAspectRatioPadRightBottom(gpuImg, m_inputH, m_inputW);
        inputImgs.emplace_back(std::move(resized));
    }

    // Convert the batch from NHWC to NCHW
    // Also apply normalization, scaling, and mean subtraction
    auto mfloat = Engine::blobFromGpuMats(inputImgs, m_subVals, m_divVals, m_normalize);
    auto *dataPointer = mfloat.ptr<void>();

    // Copy the GPU buffer to member variable so that it persists
    checkCudaErrorCode(cudaMemcpyAsync(m_deviceInput, dataPointer, m_inputCount * sizeof(float), cudaMemcpyDeviceToDevice));

    m_imgIdx += m_batchSize;
    if (std::string(names[0]) != m_inputBlobName) {
        std::cerr << "Int8EntropyCalibrator2::getBatch error: unexpected input name "
                  << names[0] << " (expected " << m_inputBlobName << ")" << std::endl;
        return false;
    }
    bindings[0] = m_deviceInput;
    return true;
}

void const *Int8EntropyCalibrator2::readCalibrationCache(size_t &length) noexcept {
    m_calibCache.clear();
    std::ifstream input(m_calibTableName, std::ios::binary);
    input >> std::noskipws;
    if (m_readCache && input.good()) {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibCache));
    }
    length = m_calibCache.size();
    return length ? m_calibCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void *ptr, std::size_t length) noexcept {
    std::ofstream output(m_calibTableName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(ptr), length);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() {
    // Log instead of throwing — destructors must not throw.
    const cudaError_t code = cudaFree(m_deviceInput);
    if (code != cudaSuccess) {
        std::cerr << "Int8EntropyCalibrator2::~Int8EntropyCalibrator2 cudaFree failed: "
                  << cudaGetErrorString(code) << std::endl;
    }
}

