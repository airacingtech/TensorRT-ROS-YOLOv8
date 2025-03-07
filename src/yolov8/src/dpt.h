#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "utils.h"
#include <cuda_runtime.h>
// #include "cuda_postprocess.h"

// Ensure CUDA function is correctly linked
#ifdef __cplusplus
extern "C" {
#endif
void launchPostprocessKernel(float* depth, int totalElements, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

class DepthAnything
{
public:
	DepthAnything();
    void init(std::string model_path, nvinfer1::ILogger& logger);
	// cv::Mat predict(cv::Mat& image, cv::Mat& depth_output, bool infer_mode=true);
	cv::Mat predict(cv::Mat& image, bool infer_mode);
	~DepthAnything();
	
private:
	int input_w = 1008;
	int input_h = 154;
	float mean[3] = { 123.675, 116.28, 103.53 };
	float std[3] = { 58.395, 57.12, 57.375 };

	std::vector<int> offset;

	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	nvinfer1::INetworkDefinition* network;

	void* buffer[2];
	float* depth_data;
	cudaStream_t stream;

	std::vector<float> preprocess(cv::Mat image);
	cv::Mat reversePreprocess(const cv::Mat& processed_image, int orig_width, int orig_height);
	std::vector<DepthEstimation> postprocess(std::vector<int> mask, int img_w, int img_h);
	void build(std::string onnxPath, nvinfer1::ILogger& logger);
	bool saveEngine(const std::string& filename);
};
