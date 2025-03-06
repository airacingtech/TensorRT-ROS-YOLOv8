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
std::vector<float> DepthAnything::preprocess(cv::Mat image)
{
    // See Cropping and Resizing
    int orig_width = image.cols;
    int orig_height = image.rows;
    double target_aspect = static_cast<double>(input_w) / input_h;
    double orig_aspect = static_cast<double>(orig_width) / orig_height;

    // Always Fully utilize the width
    int crop_width = orig_width;
    int crop_height = static_cast<int>(orig_width / target_aspect); // This needs to be memorized
    int y_offset = (orig_height - crop_height) / 2;
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

cv::Mat DepthAnything::reversePreprocess(const cv::Mat& processed_image, int orig_width, int orig_height)
{
    // The target aspect remains the same as in preprocessing.
    double target_aspect = static_cast<double>(input_w) / input_h;
    
    // The crop in the forward pass used the full width and a computed crop_height.
    int crop_width = orig_width;
    int crop_height = static_cast<int>(orig_width / target_aspect);
    
    // Compute the vertical offset (the same used for cropping in preprocessing)
    int y_offset = (orig_height - crop_height) / 2;
    
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
cv::Mat DepthAnything::predict(cv::Mat& image, bool infer_mode=true)
{
    // Original Image Dimensions
    int orig_w = image.cols; 
    int orig_h = image.rows;

    // Preprocessing
    std::vector<float> input = infer_mode ? preprocess(std::move(image)) : preprocess(image); // preprocess by default makes a copy
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
    // std::cout << "Depth map size: " << depth_mat.size() << std::endl;
    // cv::normalize(depth_mat, depth_mat, 0, 120, cv::NORM_MINMAX, CV_8U);

    // **Resize depth map to original image aspect ratio**
    // std::tuple<cv::Mat, int, int> resized = resize_depth(depth_mat, img_w, img_h);
    // cv::Mat padded_depth = std::get<0>(resized);
    depth_mat = reversePreprocess(depth_mat, orig_w, orig_h);

    // int resize_w, resize_h;
    // cv::Mat resized_depth;
    // float aspect_ratio = static_cast<float>(input_w) / input_h;

    // if (aspect_ratio > 1.0) { // image too wide
    //     resize_w = orig_w;
    //     resize_h = static_cast<int>(resize_w / aspect_ratio);
    // } else {
    //     resize_h = orig_h;
    //     resize_w = static_cast<int>(resize_h * aspect_ratio);
    // }
    // cv::resize(depth_mat, resized_depth, cv::Size(resize_w, resize_h));
    // // std::cout << "Depth RESIZE size: " << resized_depth.size() << std::endl;
    // // **Pad resized depth map to original size (centered)**
    // int pad_top = (orig_h - resize_h) / 2;
    // int pad_bottom = orig_h - resize_h - pad_top;
    // int pad_left = (orig_w - resize_w) / 2;
    // int pad_right = orig_w - resize_w - pad_left;

    // cv::Mat padded_depth;
    // cv::copyMakeBorder(resized_depth, padded_depth, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));

    return depth_mat;
}

// cv::cuda::GpuMat DepthAnything::preprocess(const cv::cuda::GpuMat &input) {
//     // Check if the input is large enough to perform the crop.
//     if (input.cols < static_cast<int>(crop_width) || input.rows < static_cast<int>(crop_height)) {
//         throw std::runtime_error("Input image is smaller than the crop dimensions.");
//     }

//     // Calculate the top-left corner for the center crop.
//     int x = (input.cols - input_w) / 2;
//     int y = (input.rows - input_h) / 2;

//     // Define the region of interest (ROI) for the center crop.
//     cv::Rect roi(x, y, input_w, input_h);

//     // Return the cropped region.
//     return input(roi);
// }

/*
* Run inference on a batch of images. Note this function only support segmentation models.
* 
* @param gpuImgs: a vector of input images
*/
std::vector<cv::Mat> DepthAnything::detectObjects(std::vector<cv::cuda::GpuMat> &gpuImgs) {

    std::vector<cv::Mat> featureVectors;
    std::vector<cv::cuda::GpuMat> input_batches;
    for (size_t i = 0; i < gpuImgs.size(); ++i) {
        input_batches.push_back(gpuImgs[i]);
        
        // Copying the images into the GPU
        // cv::cuda::GpuMat mfloat = blobFromGpuMats(input_batches, m_subVals, m_divVals, m_normalize);
        // auto *dataPointer = mfloat.ptr<void>();
        // checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], dataPointer,
        //     // mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float),
        //     blob_size,
        //     cudaMemcpyDeviceToDevice, inferenceCudaStream));
        cudaMemcpyAsync(buffer[0], input_batches.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Run inference.
        std::cout << "Running enqueueV3..." << std::endl;
        bool status = context->enqueueV3(stream);
        // if (!status) {
        //     std::cout << "===== Error =====" << std::endl;
        //     std::cout << "Error calling enqueueV3!" << std::endl;
        //     return false;
        // }

        // TODO: Launch Postprocess
        // int totalElements = input_w * input_h;
        // launchPostprocessKernel((float*) buffer[1], totalElements, stream);

        // auto succ = m_trtEngine->runInference(input, featureVectors);
    }

    // if (!status) {
    //     throw std::runtime_error("Error: Unable to run inference.");
    // }
// #ifdef ENABLE_BENCHMARKS
//     static long long t2 = 0;
//     t2 += s2.elapsedTime<long long, std::chrono::microseconds>();
//     std::cout << "Avg Inference time: " << (t2 / numIts) / 1000.f << " ms" << std::endl;
//     preciseStopwatch s3;
// #endif
//     std::vector<std::vector<Object>> ret;
//     ret = postProcessSegmentation(featureVectors);
// #ifdef ENABLE_BENCHMARKS
//     static long long t3 = 0;
//     t3 +=  s3.elapsedTime<long long, std::chrono::microseconds>();
//     std::cout << "Avg Postprocess time: " << (t3 / numIts++) / 1000.f << " ms\n" << std::endl;
// #endif
    return featureVectors;
}

/**
 * Uploads the batched input images to GPU memory and calls detectObjects(...) on the GPU images.
 * 
 * @param imgMat The batched images in BGR format.
 * @return A vector of detected objects.
 */
std::vector<cv::Mat> DepthAnything::detectObjects(std::vector<cv::Mat> &imgMats) {
    std::vector<cv::cuda::GpuMat> gpuImgs;

    // TODO: Bench Upload with CUDA streams vs sequentially
    for (const cv::Mat& img : imgMats) {
        cv::Mat clone_image;
        img.copyTo(clone_image);
        // TODO: Apply crop for DPT size
        std::vector<float> input = preprocess(clone_image);

        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(clone_image);
        gpuImgs.push_back(gpuImg);
    }
    
    // Call detectObjects with the GPU image
    return detectObjects(gpuImgs);
}


cv::Mat DepthAnything::infer(cv::Mat& image)
{
    cv::Mat clone_image;
    image.copyTo(clone_image);

    int img_w = image.cols;
    int img_h = image.rows;

    // Preprocessing
    std::vector<float> input = preprocess(clone_image);
    cudaMemcpyAsync(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Inference using depth estimation model
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2(buffer, stream, nullptr);
#else
    context->executeV2(buffer);
#endif
    // --- GPU Postprocessing ---
    // depth = 120 - torch.clamp(depth, min=0, max=120)
    int totalElements = img_h * img_w;
    launchPostprocessKernel((float*) buffer[1], totalElements, stream);
    // --------------------------

    cudaStreamSynchronize(stream);

    // Postprocessing
    cudaMemcpyAsync(depth_data, buffer[1], input_h * input_w * sizeof(float), cudaMemcpyDeviceToHost);

    return cv::Mat(input_h, input_w, CV_32FC1, depth_data).clone();  // ✅ Clone ensures ownership
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