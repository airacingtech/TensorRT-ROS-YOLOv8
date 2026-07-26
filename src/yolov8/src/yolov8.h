#pragma once
#include "engine.h"

struct Object {
    int label{};
    float probability{};
    cv::Rect_<float> rect;
    // Per-instance binary segmentation mask, in the bounding box's local coordinates.
    cv::Mat boxMask;
};

// Configures the YoloV8 detector. Values can be overridden via the CLI flags handled in cmd_line_util.h.
struct YoloV8Config {
    Precision precision = Precision::FP16;
    // Calibration data directory. Required when precision == INT8.
    std::string calibrationDataDirectory;
    // Batch size to pin the TensorRT optimization profile to. Must match the ONNX model's
    // declared batch dimension and the number of camera topics.
    int batchSize = 4;
    float probabilityThreshold = 0.25f;
    float nmsThreshold = 0.65f;
    int topK = 100;
    // Segmentation prototype dimensions; must match the ONNX model's `output1` shape.
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // Class names indexed by the model's class id. Override via --class-names. Each entry needs a
    // corresponding color in COLOR_LIST below.
    std::vector<std::string> classNames = { "car" };
};

class YoloV8 {
public:
    // Builds the ONNX model into a TensorRT engine and loads it into memory.
    YoloV8(const std::string& onnxModelPath, const YoloV8Config& config);

    // Upload a batch of host images to the GPU and run inference. Used by the ROS node.
    std::vector<std::vector<Object>> detectObjects(std::vector<cv::Mat> &imgMats);

    // Run inference on a batch of images already resident on the GPU. The ROS node does not
    // need this directly, but it is part of the public API for callers that already produce
    // GpuMats (e.g. an upstream CUDA-accelerated source).
    std::vector<std::vector<Object>> detectObjects(std::vector<cv::cuda::GpuMat> &gpuImgs);

    // Combine every detection's binary mask into a single channel image where pixel value = 1-based instance id.
    void getOneChannelSegmentationMask(const std::vector<Object>& objects, cv::Mat& segMaskOneChannel, int img_height, int img_width);

    // Draw bounding boxes, labels, and masks onto an image.
    void drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, unsigned int scale = 2);

    int getNumClasses() const { return CLASS_NAMES.size(); }
    std::string getClassName(int i) const { return CLASS_NAMES[i]; }
    int getBatchSize() const { return m_trtEngine->getBatchSize(); }
    int getInputHeight() const { return m_trtEngine->getInputDims()[0].d[1]; }
    int getInputWidth() const { return m_trtEngine->getInputDims()[0].d[2]; }

private:
    // Which output binding is the detection head and which the mask prototypes; resolved once
    // from the engine's output shapes.
    void resolveOutputBindings();
    int m_detIdx = 0;
    int m_protoIdx = 1;
    cv::cuda::GpuMat m_headT;
    cv::cuda::GpuMat m_bestScores;
    // All post-processing runs on one stream and is waited on once with a blocking event.
    // Each default-stream OpenCV CUDA call would otherwise spin-wait on its own.
    cv::cuda::Stream m_stream;
    cudaEvent_t m_ppDone = nullptr;
    void syncPostProcess();
    bool m_bindingsResolved = false;

    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(std::vector<cv::cuda::GpuMat> &gpuImgs);

    std::vector<std::vector<Object>> postProcessSegmentation(int batchSize);
    std::vector<Object> postProcessBatchItem(int batch_index);

    std::unique_ptr<Engine> m_trtEngine = nullptr;

    // YOLOv8 expects input pixels in [0, 1]; no mean subtraction or scaling beyond /255.
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    // Per-batch-element resize ratios and original image dimensions, populated by preprocess().
    std::vector<float> m_ratio_;
    std::vector<float> m_imgWidth_;
    std::vector<float> m_imgHeight_;

    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    const int SEG_CHANNELS;
    const int SEG_H;
    const int SEG_W;
    const float SEGMENTATION_THRESHOLD;

    const std::vector<std::string> CLASS_NAMES;

    // Colors used to render bounding boxes / masks. Each class is mapped via `label % COLOR_LIST.size()`.
    const std::vector<std::vector<float>> COLOR_LIST = {
            {0.098, 0.325, 0.850},
            {0.125, 0.694, 0.929},
    };
};
