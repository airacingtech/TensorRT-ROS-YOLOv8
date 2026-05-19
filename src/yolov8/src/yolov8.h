#pragma once
#include "engine.h"
#include <fstream>

// Utility method for checking if a file exists on disk
inline bool doesFileExist(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

struct Object {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Contours of semantic segmentation mask
    std::vector<std::vector<cv::Point>> contours;
    // Pose estimation key points
    std::vector<float> kps{};
};

// Config the behavior of the YoloV8 detector.
// Can pass these arguments as command line parameters.
struct YoloV8Config {
    // The precision to be used for inference
    Precision precision = Precision::FP16;
    // Calibration data directory. Must be specified when using INT8 precision.
    std::string calibrationDataDirectory;
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.65f;
    // Max number of detected objects to return
    int topK = 100;
    // Segmentation config options (these are the output dimensions of the ONNX model - from output1 in the ONNX graph)
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // Pose estimation options
    int numKPS = 17;
    float kpsThreshold = 0.5f;
    // Class names indexed by the model's class id. Override via the --class-names CLI flag or by
    // editing yolov8.env. Each entry needs a corresponding color in COLOR_LIST below.
    std::vector<std::string> classNames = { "car" };
};

class YoloV8 {
public:
    // Builds the ONNX model into a TensorRT engine and loads it into memory.
    YoloV8(const std::string& onnxModelPath, const YoloV8Config& config);

    // Upload a batch of host images to the GPU and run inference.
    std::vector<std::vector<Object>> detectObjects(std::vector<cv::Mat> &imgMats);

    // Run inference on a batch of images already resident on the GPU.
    std::vector<std::vector<Object>> detectObjects(std::vector<cv::cuda::GpuMat> &gpuImgs);

    // Combine every detection's binary mask into a single channel image where pixel value = 1-based instance id.
    void getOneChannelSegmentationMask(const std::vector<Object>& objects, cv::Mat& segMaskOneChannel, int img_height, int img_width);

    // Draw object bounding boxes, labels, and masks (if present) onto an image.
    void drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, unsigned int scale = 2);

    // Overload that also fills `masks` with the binary mask of each detected instance.
    void drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, std::vector<cv::Mat> &masks, unsigned int scale = 2);

    int getNumClasses() const { return CLASS_NAMES.size(); }
    std::string getClassName(int i) const { return CLASS_NAMES[i]; }
    int getBatchSize() const { return m_trtEngine->getBatchSize(); }
    int getInputHeight() const { return m_trtEngine->getInputDims()[0].d[1]; }
    int getInputWidth() const { return m_trtEngine->getInputDims()[0].d[2]; }

private:
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat& gpuImg);
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(std::vector<cv::cuda::GpuMat> &gpuImgs);

    std::vector<std::vector<Object>> postProcessSegmentation(std::vector<std::vector<std::vector<float>>>& batchedFeatureVectors);
    std::vector<Object> postProcessSegmentation(std::vector<std::vector<float>>& featureVectors, int batch_index);

    std::unique_ptr<Engine> m_trtEngine = nullptr;

    // YOLOv8 expects input pixels in [0, 1]; SUB/DIV/NORMALIZE leave defaults.
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    // Per-batch-element resize ratios and original image dimensions, populated by preprocess().
    std::vector<float> m_ratio_;
    std::vector<float> m_imgWidth_;
    std::vector<float> m_imgHeight_;

    // Filter thresholds
    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    // Segmentation constants
    const int SEG_CHANNELS;
    const int SEG_H;
    const int SEG_W;
    const float SEGMENTATION_THRESHOLD;

    // Object classes as strings
    const std::vector<std::string> CLASS_NAMES;

    // Pose estimation constant
    const int NUM_KPS;
    const float KPS_THRESHOLD;

    // Color list for drawing objects
    const std::vector<std::vector<float>> COLOR_LIST = {
            {0.098, 0.325, 0.850},
            {0.125, 0.694, 0.929},
    };

    const std::vector<std::vector<unsigned int>> KPS_COLORS = {
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255}
    };

    const std::vector<std::vector<unsigned int>> SKELETON = {
            {16, 14},
            {14, 12},
            {17, 15},
            {15, 13},
            {12, 13},
            {6, 12},
            {7, 13},
            {6, 7},
            {6, 8},
            {7, 9},
            {8, 10},
            {9, 11},
            {2, 3},
            {1, 2},
            {1, 3},
            {2, 4},
            {3, 5},
            {4, 6},
            {5, 7}
    };

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {255, 51, 255},
            {255, 51, 255},
            {255, 51, 255},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0}
    };
};