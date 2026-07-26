#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "yolov8.h"

YoloV8::YoloV8(const std::string& onnxModelPath, const YoloV8Config& config)
        : PROBABILITY_THRESHOLD(config.probabilityThreshold)
        , NMS_THRESHOLD(config.nmsThreshold)
        , TOP_K(config.topK)
        , SEG_CHANNELS(config.segChannels)
        , SEG_H(config.segH)
        , SEG_W(config.segW)
        , SEGMENTATION_THRESHOLD(config.segmentationThreshold)
        , CLASS_NAMES(config.classNames) {
    if (config.batchSize <= 0) {
        throw std::runtime_error("Error: batch size must be > 0 (got " + std::to_string(config.batchSize) + ")");
    }

    Options options;
    options.optBatchSize = config.batchSize;
    options.maxBatchSize = config.batchSize;

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

std::vector<std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(std::vector<cv::cuda::GpuMat> &gpuImgs) {
    const std::vector<nvinfer1::Dims3>& inputDims = m_trtEngine->getInputDims();
    // The TensorRT engine input is shaped [input][batch][cv::cuda::GpuMat].
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    std::vector<cv::cuda::GpuMat> batches;
    m_imgHeight_.clear();
    m_imgWidth_.clear();
    m_ratio_.clear();
    size_t slot = 0;
    for (cv::cuda::GpuMat &gpuImg : gpuImgs) {
        // Cache the original dimensions and the resize ratio so detections can be mapped back to
        // the source image resolution during post-processing.
        m_imgHeight_.push_back(gpuImg.rows);
        m_imgWidth_.push_back(gpuImg.cols);
        m_ratio_.push_back(1.f / std::min(inputDims[0].d[2] / static_cast<float>(gpuImg.cols),
            inputDims[0].d[1] / static_cast<float>(gpuImg.rows)));
        // Letterbox to the engine input size if the source image doesn't already match.
        if (gpuImg.rows != inputDims[0].d[1] || gpuImg.cols != inputDims[0].d[2]) {
            gpuImg = m_trtEngine->letterboxInto(gpuImg, slot, inputDims[0].d[1], inputDims[0].d[2]);
        }
        ++slot;
        batches.push_back(std::move(gpuImg));
    }
    inputs.push_back(std::move(batches));
    return inputs;
}

void YoloV8::syncPostProcess() {
    if (m_ppDone == nullptr) {
        cudaEventCreateWithFlags(&m_ppDone, cudaEventBlockingSync | cudaEventDisableTiming);
    }
    cudaEventRecord(m_ppDone, cv::cuda::StreamAccessor::getStream(m_stream));
    cudaEventSynchronize(m_ppDone);
}

void YoloV8::resolveOutputBindings() {
    if (m_bindingsResolved) {
        return;
    }
    // Binding order depends on the ONNX exporter, so identify each output by shape: the
    // detection head has more rows than the mask channels, the prototypes fewer.
    const auto& outputDims = m_trtEngine->getOutputDims();
    for (size_t k = 0; k < outputDims.size(); ++k) {
        if (outputDims[k].d[1] > SEG_CHANNELS + 4) {
            m_detIdx = static_cast<int>(k);
        } else {
            m_protoIdx = static_cast<int>(k);
        }
    }
    m_bindingsResolved = true;
}

std::vector<std::vector<Object>> YoloV8::detectObjects(std::vector<cv::cuda::GpuMat> &gpuImgs) {
    resolveOutputBindings();
    const auto input = preprocess(gpuImgs);
    std::vector<std::vector<std::vector<float>>> featureVectors;

    // The mask prototypes are the largest output by far and are only ever consumed by the GPU
    // mask pipeline below, so they stay resident on the device.
    // Both outputs are consumed on the device, so nothing is copied back from inference.
    const std::vector<bool> download(m_trtEngine->getOutputDims().size(), false);

    if (!m_trtEngine->runInference(input, featureVectors, download)) {
        throw std::runtime_error("Error: Unable to run inference.");
    }
    return postProcessSegmentation(static_cast<int>(gpuImgs.size()));
}

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

std::vector<std::vector<Object>> YoloV8::postProcessSegmentation(int batchSize) {
    std::vector<std::vector<Object>> batched_objects;
    batched_objects.reserve(batchSize);
    for (int batch_index = 0; batch_index < batchSize; ++batch_index) {
        batched_objects.push_back(postProcessBatchItem(batch_index));
    }
    return batched_objects;
}

std::vector<Object> YoloV8::postProcessBatchItem(int batch_index) {
    const auto& outputDims = m_trtEngine->getOutputDims();
    const int detIdx = m_detIdx;

    const int numChannels = outputDims[detIdx].d[1];
    const int numAnchors = outputDims[detIdx].d[2];
    const int numClasses = numChannels - SEG_CHANNELS - 4;

    // The detection head stays on the device. Transposing there also replaces a full-size
    // host transpose of the [numChannels, numAnchors] block.
    const cv::cuda::GpuMat head(numChannels, numAnchors, CV_32F,
                                m_trtEngine->getOutputDevicePtr(detIdx, batch_index));
    cv::cuda::transpose(head, m_headT, m_stream);

    const cv::cuda::GpuMat classBlock = m_headT.colRange(4, 4 + numClasses);
    if (numClasses == 1) {
        m_bestScores = classBlock;
    } else {
        cv::cuda::reduce(classBlock, m_bestScores, 1, cv::REDUCE_MAX, CV_32F, m_stream);
    }

    // One 34 KB copy of the per-anchor scores, then a single wait. Scanning them on the host
    // is a few microseconds and replaces a separate minMaxLoc round trip.
    cv::Mat bestHost;
    m_bestScores.download(bestHost, m_stream);
    syncPostProcess();

    std::vector<int> keep;
    keep.reserve(64);
    for (int i = 0; i < numAnchors; ++i) {
        if (bestHost.at<float>(i) > PROBABILITY_THRESHOLD) {
            keep.push_back(i);
        }
    }
    if (keep.empty()) {
        return {};
    }

    // Gather the surviving anchors on the device so only they cross back, instead of the
    // whole head.
    cv::cuda::GpuMat gathered(static_cast<int>(keep.size()), numChannels, CV_32F);
    for (size_t k = 0; k < keep.size(); ++k) {
        m_headT.row(keep[k]).copyTo(gathered.row(static_cast<int>(k)), m_stream);
    }
    cv::Mat survivors;
    gathered.download(survivors, m_stream);
    syncPostProcess();

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> maskConfs;
    std::vector<int> indices;

    for (int i = 0; i < survivors.rows; i++) {
        auto rowPtr = survivors.ptr<float>(i);
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maskConfsPtr = rowPtr + 4 + numClasses;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        {
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

            cv::Mat maskConf = cv::Mat(1, SEG_CHANNELS, CV_32F, maskConfsPtr).clone();

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

    // Build the per-instance segmentation masks on the GPU. The prototypes were never copied
    // back, so the matmul, the upsample and the threshold all run on the device and only each
    // object's own box-sized mask crosses back to the host.
    if (!masks.empty()) {
        cv::cuda::GpuMat masksGpu;
        masksGpu.upload(masks, m_stream);
        const cv::cuda::GpuMat protosGpu(
            SEG_CHANNELS, SEG_H * SEG_W, CV_32F,
            m_trtEngine->getOutputDevicePtr(m_protoIdx, batch_index));

        cv::cuda::GpuMat scores;
        cv::cuda::gemm(masksGpu, protosGpu, 1.0, cv::cuda::GpuMat(), 0.0, scores, 0, m_stream);

        // sigmoid(x) > t is exactly x > logit(t), so the sigmoid pass is unnecessary.
        const float t = std::clamp(SEGMENTATION_THRESHOLD, 1e-6f, 1.f - 1e-6f);
        const double logitThreshold = std::log(t / (1.f - t));

        cv::Rect roi;
        if (m_imgHeight_[batch_index] > m_imgWidth_[batch_index]) {
            roi = cv::Rect(0, 0, SEG_W * m_imgWidth_[batch_index] / m_imgHeight_[batch_index], SEG_H);
        } else {
            roi = cv::Rect(0, 0, SEG_W, SEG_H * m_imgHeight_[batch_index] / m_imgWidth_[batch_index]);
        }

        // Upsample only the slice of the prototype that the box covers, straight to box size.
        // Resizing the full frame and cropping afterwards does ~380x more work per detection
        // for the same interior pixels -- bilinear sampling is local, so this is equivalent.
        const float sx = static_cast<float>(roi.width) / m_imgWidth_[batch_index];
        const float sy = static_cast<float>(roi.height) / m_imgHeight_[batch_index];
        cv::cuda::GpuMat boxF, boxThresh;
        for (size_t i = 0; i < objs.size(); i++) {
            const cv::Rect& r = objs[i].rect;
            if (r.width <= 0 || r.height <= 0) {
                continue;
            }
            cv::Rect src(static_cast<int>(std::floor(r.x * sx)),
                         static_cast<int>(std::floor(r.y * sy)),
                         std::max(1, static_cast<int>(std::ceil(r.width * sx))),
                         std::max(1, static_cast<int>(std::ceil(r.height * sy))));
            src &= cv::Rect(0, 0, roi.width, roi.height);
            if (src.width <= 0 || src.height <= 0) {
                continue;
            }

            const cv::cuda::GpuMat proto(SEG_H, SEG_W, CV_32F, scores.ptr<float>(i));
            cv::cuda::resize(proto(roi)(src), boxF, r.size(), 0, 0, cv::INTER_LINEAR, m_stream);
            cv::cuda::threshold(boxF, boxThresh, logitThreshold, 255.0, cv::THRESH_BINARY, m_stream);
            boxThresh.convertTo(boxThresh, CV_8U, m_stream);
            boxThresh.download(objs[i].boxMask, m_stream);
            syncPostProcess();
        }
    }

    return objs;
}

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
                                     "Update CLASS_NAMES in yolov8.env (or --class-names) to match the model.");
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
    }
}

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
