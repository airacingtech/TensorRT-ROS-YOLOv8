#include <opencv2/cudaimgproc.hpp>
#include "yolov8.h"

YoloV8::YoloV8(const std::string& onnxModelPath, const YoloV8Config& config)
        : PROBABILITY_THRESHOLD(config.probabilityThreshold)
        , NMS_THRESHOLD(config.nmsThreshold)
        , TOP_K(config.topK)
        , SEG_CHANNELS(config.segChannels)
        , SEG_H(config.segH)
        , SEG_W(config.segW)
        , SEGMENTATION_THRESHOLD(config.segmentationThreshold)
        , CLASS_NAMES(config.classNames)
        , NUM_KPS(config.numKPS)
        , KPS_THRESHOLD(config.kpsThreshold) {
    // Specify options for GPU inference
    Options options;
    options.optBatchSize = 4;
    options.maxBatchSize = 4;

    options.precision = config.precision;
    options.calibrationDataDirectoryPath = config.calibrationDataDirectory;

    if (options.precision == Precision::INT8 && options.calibrationDataDirectoryPath.empty()) {
        throw std::runtime_error("Error: Must supply calibration data path for INT8 calibration");
    }

    m_trtEngine = std::make_unique<Engine>(options);

    // Build the ONNX model into a TensorRT engine file. If a matching engine is already cached
    // on disk this returns immediately. The cache key includes the build options, so any change
    // to Options above triggers a rebuild.
    if (!m_trtEngine->build(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE)) {
        throw std::runtime_error("Error: Unable to build the TensorRT engine. "
                                 "Try raising the TensorRT logger to kVERBOSE in engine.cpp for details.");
    }

    if (!m_trtEngine->loadNetwork(onnxModelPath)) {
        throw std::runtime_error("Error: Unable to load TensorRT engine weights into memory.");
    }
}

/*
 * Preprocess a batch of images for inference on the TensorRT Engine.
 *
 * @param gpuImgs A batch of input images to preprocess
 */
std::vector<std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(std::vector<cv::cuda::GpuMat> &gpuImgs) {
    const std::vector<nvinfer1::Dims3>& inputDims = m_trtEngine->getInputDims();
    // The input to the neural network is a tensor with dimensions [input][batch][cv::cuda::GpuMat].
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    std::vector<cv::cuda::GpuMat> batches;
    m_imgHeight_.clear();
    m_imgWidth_.clear();
    m_ratio_.clear();
    for (cv::cuda::GpuMat &gpuImg : gpuImgs) {
        // Store the original image dimensions and the ratio used to resize the image so we
        // can convert detections back to the original image size during post-processing.
        m_imgHeight_.push_back(gpuImg.rows);
        m_imgWidth_.push_back(gpuImg.cols);
        m_ratio_.push_back(1.f / std::min(inputDims[0].d[2] / static_cast<float>(gpuImg.cols),
            inputDims[0].d[1] / static_cast<float>(gpuImg.rows)));
        // Resize to the model expected input size while maintaining the aspect ratio with padding.
        if (gpuImg.rows != inputDims[0].d[1] || gpuImg.cols != inputDims[0].d[2]) {
            gpuImg = Engine::resizeKeepAspectRatioPadRightBottom(gpuImg, inputDims[0].d[1], inputDims[0].d[2]);
        }
        batches.push_back(std::move(gpuImg));
    }
    inputs.push_back(std::move(batches));
    return inputs;
}

std::vector<std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(const cv::cuda::GpuMat &gpuImg) {
    const auto& inputDims = m_trtEngine->getInputDims();
    auto resized = gpuImg;
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        resized = Engine::resizeKeepAspectRatioPadRightBottom(gpuImg, inputDims[0].d[1], inputDims[0].d[2]);
    }

    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

    m_imgHeight_.push_back(gpuImg.rows);
    m_imgWidth_.push_back(gpuImg.cols);
    m_ratio_.push_back(1.f / std::min(inputDims[0].d[2] / static_cast<float>(gpuImg.cols),
                                      inputDims[0].d[1] / static_cast<float>(gpuImg.rows)));
    return inputs;
}

/*
 * Run inference on a batch of images. Supports only segmentation models.
 *
 * @param gpuImgs A vector of input images
 */
std::vector<std::vector<Object>> YoloV8::detectObjects(std::vector<cv::cuda::GpuMat> &gpuImgs) {
#ifdef ENABLE_BENCHMARKS
    static int numIts = 1;
    preciseStopwatch s1;
#endif
    const auto input = preprocess(gpuImgs);
#ifdef ENABLE_BENCHMARKS
    static long long t1 = 0;
    t1 += s1.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Preprocess time: " << (t1 / numIts) / 1000.f << " ms" << std::endl;
    preciseStopwatch s2;
#endif
    std::vector<std::vector<std::vector<float>>> featureVectors;
    if (!m_trtEngine->runInference(input, featureVectors)) {
        throw std::runtime_error("Error: Unable to run inference.");
    }
#ifdef ENABLE_BENCHMARKS
    static long long t2 = 0;
    t2 += s2.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Inference time: " << (t2 / numIts) / 1000.f << " ms" << std::endl;
    preciseStopwatch s3;
#endif
    auto ret = postProcessSegmentation(featureVectors);
#ifdef ENABLE_BENCHMARKS
    static long long t3 = 0;
    t3 += s3.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Postprocess time: " << (t3 / numIts++) / 1000.f << " ms\n" << std::endl;
#endif
    return ret;
}

/**
 * Uploads the batched input images to GPU memory and calls detectObjects(...) on the GPU images.
 *
 * @param imgMats The batched images in BGR format.
 * @return A vector of detected objects.
 */
std::vector<std::vector<Object>> YoloV8::detectObjects(std::vector<cv::Mat> &imgMats) {
    std::vector<cv::cuda::GpuMat> gpuImgs;
    gpuImgs.reserve(imgMats.size());
    for (const cv::Mat& img : imgMats) {
        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(img);
        gpuImgs.push_back(std::move(gpuImg));
    }
    return detectObjects(gpuImgs);
}

/**
 * Performs post-processing on a batch of segmentation feature vectors.
 *
 * @param batchedFeatureVectors Batched vector of size [batch][output][feature_vector].
 * @return A batch of vectors of Object instances.
 */
std::vector<std::vector<Object>> YoloV8::postProcessSegmentation(std::vector<std::vector<std::vector<float>>>& batchedFeatureVectors) {
    std::vector<std::vector<Object>> batched_objects;
    int batch_index = 0;
    for (std::vector<std::vector<float>> &featureVectors : batchedFeatureVectors) {
        batched_objects.push_back(postProcessSegmentation(featureVectors, batch_index));
        batch_index++;
    }
    return batched_objects;
}

/**
 * Performs post-processing on the two output buffers of a single batch element.
 *
 * @param featureVectors The 2D feature vectors to be processed.
 * @param batch_index The index of the batch.
 * @return A vector of Object instances for this batch element.
 */
std::vector<Object> YoloV8::postProcessSegmentation(std::vector<std::vector<float>>& featureVectors, int batch_index) {
    const auto& outputDims = m_trtEngine->getOutputDims();

    const int numChannels = outputDims[outputDims.size() - 1].d[1];
    const int numAnchors = outputDims[outputDims.size() - 1].d[2];
    const int numClasses = numChannels - SEG_CHANNELS - 4;

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVectors[1].data());
    output = output.t();

    cv::Mat protos = cv::Mat(SEG_CHANNELS, SEG_H * SEG_W, CV_32F, featureVectors[0].data());

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> maskConfs;
    std::vector<int> indices;

    for (int i = 0; i < numAnchors; i++) {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maskConfsPtr = rowPtr + 4 + numClasses;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > PROBABILITY_THRESHOLD) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratio_[batch_index], 0.f, m_imgWidth_[batch_index]);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio_[batch_index], 0.f, m_imgHeight_[batch_index]);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio_[batch_index], 0.f, m_imgWidth_[batch_index]);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio_[batch_index], 0.f, m_imgHeight_[batch_index]);

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            cv::Mat maskConf = cv::Mat(1, SEG_CHANNELS, CV_32F, maskConfsPtr);

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            maskConfs.push_back(maskConf);
        }
    }

    // Requires OpenCV 4.7+.
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

    cv::Mat masks;
    std::vector<Object> objs;
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= TOP_K) {
            break;
        }
        Object obj;
        obj.label = labels[i];
        obj.rect = bboxes[i];
        obj.probability = scores[i];
        masks.push_back(maskConfs[i]);
        objs.push_back(obj);
        cnt++;
    }

    // Build the per-instance segmentation masks at original image resolution.
    if (!masks.empty()) {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), { SEG_W, SEG_H });

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);

        cv::Rect roi;
        if (m_imgHeight_[batch_index] > m_imgWidth_[batch_index]) {
            roi = cv::Rect(0, 0, SEG_W * m_imgWidth_[batch_index] / m_imgHeight_[batch_index], SEG_H);
        } else {
            roi = cv::Rect(0, 0, SEG_W, SEG_H * m_imgHeight_[batch_index] / m_imgWidth_[batch_index]);
        }

        for (size_t i = 0; i < indices.size(); i++) {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask,
                       cv::Size(static_cast<int>(m_imgWidth_[batch_index]),
                                static_cast<int>(m_imgHeight_[batch_index])),
                       cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > SEGMENTATION_THRESHOLD;
        }
    }

    return objs;
}

/**
 * Draws bounding boxes, labels, and (if present) masks for each detection onto the image.
 */
void YoloV8::drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, unsigned int scale) {
    // Compose segmentation masks first so labels render on top.
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto& object : objects) {
            int colorIndex = object.label % COLOR_LIST.size();
            cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                          COLOR_LIST[colorIndex][1],
                                          COLOR_LIST[colorIndex][2]);
            mask(object.rect).setTo(color * 255, object.boxMask);
        }
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    for (const auto & object : objects) {
        int colorIndex = object.label % COLOR_LIST.size();
        cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                      COLOR_LIST[colorIndex][1],
                                      COLOR_LIST[colorIndex][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor = (meanColor > 0.5) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        if (object.label + 1 > getNumClasses()) {
            throw std::runtime_error("Error: detected label index exceeds the configured number of classes. "
                                     "Update yolov8.env CLASS_NAMES to match the model.");
        }

        char text[256];
        snprintf(text, sizeof(text), "%s %.1f%%", CLASS_NAMES[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, object.rect, color * 255, scale + 1);
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                      txt_bk_color, -1);
        cv::putText(image, text, cv::Point(x, y + labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

        // Pose estimation overlay (only present if the model produces keypoints).
        if (!object.kps.empty()) {
            const auto& kps = object.kps;
            for (int k = 0; k < NUM_KPS + 2; k++) {
                if (k < NUM_KPS) {
                    int kpsX = std::round(kps[k * 3]);
                    int kpsY = std::round(kps[k * 3 + 1]);
                    float kpsS = kps[k * 3 + 2];
                    if (kpsS > KPS_THRESHOLD) {
                        cv::Scalar kpsColor = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                        cv::circle(image, {kpsX, kpsY}, 5, kpsColor, -1);
                    }
                }
                const auto& ske = SKELETON[k];
                int pos1X = std::round(kps[(ske[0] - 1) * 3]);
                int pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);
                int pos2X = std::round(kps[(ske[1] - 1) * 3]);
                int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);
                float pos1S = kps[(ske[0] - 1) * 3 + 2];
                float pos2S = kps[(ske[1] - 1) * 3 + 2];

                if (pos1S > KPS_THRESHOLD && pos2S > KPS_THRESHOLD) {
                    cv::Scalar limbColor = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                    cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
                }
            }
        }
    }
}

/**
 * Composes a single-channel image where each detected instance's mask is set to its 1-based index.
 */
void YoloV8::getOneChannelSegmentationMask(const std::vector<Object>& objects, cv::Mat& segMaskOneChannel, int img_height, int img_width) {
    segMaskOneChannel = cv::Mat::zeros(img_height, img_width, CV_8UC1);
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        int i = 1;
        for (const auto& object : objects) {
            segMaskOneChannel(object.rect).setTo(i, object.boxMask);
            i++;
        }
    }
}

/**
 * Overload of drawObjectLabels that additionally returns a per-instance binary mask vector.
 */
void YoloV8::drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, std::vector<cv::Mat>& masks, unsigned int scale) {
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto& object : objects) {
            int colorIndex = object.label % COLOR_LIST.size();
            cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                          COLOR_LIST[colorIndex][1],
                                          COLOR_LIST[colorIndex][2]);
            mask(object.rect).setTo(color * 255, object.boxMask);

            cv::Mat binaryMask = cv::Mat::zeros(image.size(), CV_8UC3);
            binaryMask(object.rect).setTo(1, object.boxMask);
            masks.push_back(binaryMask);
        }
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    for (const auto & object : objects) {
        int colorIndex = object.label % COLOR_LIST.size();
        cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                      COLOR_LIST[colorIndex][1],
                                      COLOR_LIST[colorIndex][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor = (meanColor > 0.5) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        char text[256];
        snprintf(text, sizeof(text), "%s %.1f%%", CLASS_NAMES[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;
        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, object.rect, color * 255, scale + 1);
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                      txt_bk_color, -1);
        cv::putText(image, text, cv::Point(x, y + labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

        if (!object.kps.empty()) {
            const auto& kps = object.kps;
            for (int k = 0; k < NUM_KPS + 2; k++) {
                if (k < NUM_KPS) {
                    int kpsX = std::round(kps[k * 3]);
                    int kpsY = std::round(kps[k * 3 + 1]);
                    float kpsS = kps[k * 3 + 2];
                    if (kpsS > KPS_THRESHOLD) {
                        cv::Scalar kpsColor = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                        cv::circle(image, {kpsX, kpsY}, 5, kpsColor, -1);
                    }
                }
                const auto& ske = SKELETON[k];
                int pos1X = std::round(kps[(ske[0] - 1) * 3]);
                int pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);
                int pos2X = std::round(kps[(ske[1] - 1) * 3]);
                int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);
                float pos1S = kps[(ske[0] - 1) * 3 + 2];
                float pos2S = kps[(ske[1] - 1) * 3 + 2];

                if (pos1S > KPS_THRESHOLD && pos2S > KPS_THRESHOLD) {
                    cv::Scalar limbColor = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                    cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
                }
            }
        }
    }
}
