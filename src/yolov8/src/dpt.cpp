#include "dpt.h"
#include <NvOnnxParser.h>

#define isFP16 true

using namespace nvinfer1;

/**
 * @brief DepthAnything`s constructor
 * @param model_path DepthAnything engine file path
 * @param logger Nvinfer ILogger
*/
DepthAnything::DepthAnything()
{}

void DepthAnything::init(std::string model_path, nvinfer1::ILogger& logger)
{
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos)
    {
        // read the engine file
        std::ifstream engineStream(model_path, std::ios::binary);
        engineStream.seekg(0, std::ios::end);
        const size_t modelSize = engineStream.tellg();
        engineStream.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engineData(new char[modelSize]);
        engineStream.read(engineData.get(), modelSize);
        engineStream.close();

        // create tensorrt model
        runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
        context = engine->createExecutionContext();

    }
    // Build an engine from an onnx model
    else
    {
        build(model_path, logger);
        saveEngine(model_path);
    }

#if NV_TENSORRT_MAJOR < 10
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif

    // create CUDA stream
    cudaStreamCreate(&stream);
    cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float));
    cudaMalloc(&buffer[1], input_h * input_w * sizeof(float));

    depth_data = new float[input_h * input_w];
}

/**
 * @brief RTMSeg`s destructor
*/
DepthAnything::~DepthAnything()
{
    cudaFree(stream);
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);

    delete[] depth_data;
}

/**
 * @brief Network preprocessing function
 * @param image Input image
 * @return Processed Tensor
*/
std::vector<float> DepthAnything::preprocess(cv::Mat image, int upward_shift=0)
{
    // See Cropping and Resizing
    int orig_width = image.cols;
    int orig_height = image.rows;
    double target_aspect = static_cast<double>(input_w) / input_h;
    double orig_aspect = static_cast<double>(orig_width) / orig_height;

    // Always Fully utilize the width
    int crop_width = orig_width;
    int crop_height = static_cast<int>(orig_width / target_aspect); // This needs to be memorized
    upward_shift = (upward_shift + target_aspect - 1) / target_aspect; // ceil
    int y_offset = ((orig_height - crop_height) / 2) - upward_shift;
    image = image(cv::Rect(0, y_offset, crop_width, crop_height));

    // std::tuple<cv::Mat, int, int> resized = resize_depth(image, input_w, input_h);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_w, input_h));
    // Preprocessing TODO: Perform loop ordering and reserve space
    std::vector<float> input_tensor;
    // input_tensor.reserve(3 * input_h * input_w);

    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < resized_image.rows; i++)
        {
            for (int j = 0; j < resized_image.cols; j++)
            {
                input_tensor.emplace_back(((float)resized_image.at<cv::Vec3b>(i, j)[k] - mean[k]) / std[k]);
            }
        }
    }

    // // Example of better loop ordering:
    // for (int i = 0; i < resized_image.rows; i++) {
    //     const cv::Vec3b* row_ptr = resized_image.ptr<cv::Vec3b>(i);
    //     for (int j = 0; j < resized_image.cols; j++) {
    //         for (int k = 0; k < 3; k++) {
    //             input_tensor.emplace_back((static_cast<float>(row_ptr[j][k]) - mean[k]) / std[k]);
    //         }
    //     }
    // }
    return input_tensor;
}

cv::Mat DepthAnything::reversePreprocess(const cv::Mat& processed_image, int orig_width, int orig_height, int upward_shift=0)
{
    // The target aspect remains the same as in preprocessing.
    double target_aspect = static_cast<double>(input_w) / input_h;
    
    // The crop in the forward pass used the full width and a computed crop_height.
    int crop_width = orig_width;
    int crop_height = static_cast<int>(orig_width / target_aspect);
    
    // Compute the vertical offset (the same used for cropping in preprocessing)
    upward_shift = (upward_shift + target_aspect - 1) / target_aspect; // ceil
    int y_offset = ((orig_height - crop_height) / 2) - upward_shift;
    
    // Resize the processed image back to the original cropped dimensions.
    cv::Mat upscaled;
    cv::resize(processed_image, upscaled, cv::Size(crop_width, crop_height));
    
    // Create an output image filled with black (all zeros)
    cv::Mat output = cv::Mat::zeros(orig_height, orig_width, processed_image.type());
    
    // Place the upscaled image into the correct location
    upscaled.copyTo(output(cv::Rect(0, y_offset, crop_width, crop_height)));
    
    return output;
}

// cv::Mat DepthAnything::predict(cv::Mat& image, cv::Mat& depth_output, bool infer_mode=true)
cv::Mat DepthAnything::predict(cv::Mat& image, bool infer_mode=true, int upward_shift=0)
{
    // Original Image Dimensions
    int orig_w = image.cols; 
    int orig_h = image.rows;

    // Preprocessing
    std::vector<float> input = infer_mode ? preprocess(std::move(image), upward_shift) : preprocess(image, upward_shift); // preprocess by default makes a copy
    cudaMemcpyAsync(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Inference using depth estimation model
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2(buffer, stream, nullptr);
#else
    context->executeV2(buffer);
#endif
    int totalElements = input_h * input_w;
    launchPostprocessKernel((float*) buffer[1], totalElements, stream);

    cudaStreamSynchronize(stream);

    // Postprocessing
    cudaMemcpyAsync(depth_data, buffer[1], input_h * input_w * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert the entire depth_data vector to a CV_32FC1 Mat
    cv::Mat depth_mat(input_h, input_w, CV_32FC1, depth_data);
    depth_mat = reversePreprocess(depth_mat, orig_w, orig_h, upward_shift);
    return depth_mat;
}

void DepthAnything::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig* config = builder->createBuilderConfig();
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    runtime = createInferRuntime(logger);

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    context = engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool DepthAnything::saveEngine(const std::string& onnxpath)
{
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else
    {
        return false;
    }

    // Save the engine to the path
    if (engine)
    {
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}