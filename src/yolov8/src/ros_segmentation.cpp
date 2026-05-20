#include <mutex>

#include "yolov8.h"
#include "cmd_line_util.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"

// Generated ROS 2 message headers (rosidl produces snake_case file names).
#include "yolov8_interfaces/msg/point2_d.hpp"
#include "yolov8_interfaces/msg/yolov8_detections.hpp"
#include "yolov8_interfaces/msg/yolov8_b_box.hpp"

class YoloV8Node : public rclcpp::Node
{
public:
    explicit YoloV8Node(YoloV8& yoloV8)
    : Node("yolo_v8"), yoloV8_(yoloV8)
    {
        // Declare and read ROS parameters.
        this->declare_parameter<std::vector<std::string>>("camera_topics", camera_topics_);
        this->get_parameter("camera_topics", camera_topics_);
        this->declare_parameter<std::string>("camera_topic_suffix", camera_topic_suffix_);
        this->get_parameter("camera_topic_suffix", camera_topic_suffix_);
        this->declare_parameter<double>("camera_buffer_hz", camera_buffer_hz_);
        this->get_parameter("camera_buffer_hz", camera_buffer_hz_);
        this->declare_parameter<bool>("visualize_masks", visualize_masks_);
        this->get_parameter("visualize_masks", visualize_masks_);
        this->declare_parameter<bool>("enable_one_channel_mask", enable_one_channel_mask_);
        this->get_parameter("enable_one_channel_mask", enable_one_channel_mask_);
        this->declare_parameter<bool>("visualize_one_channel_mask", visualize_one_channel_mask_);
        this->get_parameter("visualize_one_channel_mask", visualize_one_channel_mask_);

        if (camera_topics_.empty()) {
            throw std::runtime_error("ROS parameter 'camera_topics' must not be empty.");
        }
        if (camera_buffer_hz_ <= 0.0) {
            throw std::runtime_error("ROS parameter 'camera_buffer_hz' must be > 0.");
        }
        if (camera_topics_.size() != static_cast<size_t>(yoloV8_.getBatchSize())) {
            throw std::runtime_error("Model batch size (" + std::to_string(yoloV8_.getBatchSize()) +
                ") does not match the number of camera topics (" + std::to_string(camera_topics_.size()) + ")");
        }
        model_input_height_ = yoloV8_.getInputHeight();
        model_input_width_ = yoloV8_.getInputWidth();

        // Timer flushes the buffer to inference if not all topics produce a frame within one period.
        const auto buffer_period = std::chrono::duration<double>(1.0 / camera_buffer_hz_);
        buffer_timer_ = rclcpp::create_wall_timer(
            buffer_period,
            std::bind(&YoloV8Node::batchBufferCallback, this),
            nullptr,
            this->get_node_base_interface().get(),
            this->get_node_timers_interface().get());

        RCLCPP_INFO(this->get_logger(), "Subscribing to %zu camera topic(s):", camera_topics_.size());
        // best_effort + volatile durability matches typical camera publishers; intra-process
        // comm avoids a serialize/deserialize round-trip when the camera driver runs in the same
        // process. None of these are subscriber-side requirements — they're paired with the
        // publishers we expect (e.g. vimba_ros2).
        rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 10));
        qos_profile.best_effort();
        qos_profile.durability_volatile();
        rclcpp::SubscriptionOptions sub_options;
        sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
        for (const std::string& topic : camera_topics_) {
            const std::string full_topic = topic + camera_topic_suffix_;
            auto subscription = this->create_subscription<sensor_msgs::msg::Image>(
                full_topic, qos_profile,
                [this, topic](const sensor_msgs::msg::Image::SharedPtr msg) {
                    this->addToBufferCallback(msg, topic);
                },
                sub_options);
            RCLCPP_INFO(this->get_logger(), "  %s", full_topic.c_str());

            subscriptions_.push_back(subscription);
            detection_publishers_[topic] = this->create_publisher<yolov8_interfaces::msg::Yolov8Detections>(
                "/yolov8" + topic + "/detections", qos_profile);
            if (visualize_masks_) {
                image_publishers_[topic] = this->create_publisher<sensor_msgs::msg::Image>(
                    "/yolov8" + topic + "/image", qos_profile);
            }
            if (enable_one_channel_mask_ && visualize_one_channel_mask_) {
                one_channel_mask_publishers_[topic] = this->create_publisher<sensor_msgs::msg::Image>(
                    "/yolov8" + topic + "/seg_mask_one_channel", qos_profile);
            }
        }
    }

private:
    using ImageBuffer = std::map<std::string, sensor_msgs::msg::Image::SharedPtr>;

    // Push a freshly arrived image into the current buffer. Older images on the same topic are
    // discarded because the network is the throughput bottleneck. If every topic has produced an
    // image, we drain the buffer immediately instead of waiting for the timer.
    void addToBufferCallback(const sensor_msgs::msg::Image::SharedPtr& image_msg, const std::string& topic) {
        std::unique_lock<std::mutex> lock(buffer_mutex_);
        current_buffer_[topic] = image_msg;
        if (current_buffer_.size() == camera_topics_.size()) {
            lock.unlock();
            batchBufferCallback();
        }
    }

    void batchBufferCallback() {
        // Move the current buffer into a local processing buffer so the subscribers can keep
        // filling and so we don't need a lock across the (slow) inference step.
        ImageBuffer processing_buffer;
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            processing_buffer.swap(current_buffer_);
        }

        buffer_timer_->reset();

        if (processing_buffer.empty()) {
            return;
        }

        // Build inputs in camera_topics_ order so the batch index trivially maps back to the topic.
        std::vector<cv::Mat> images;
        images.reserve(camera_topics_.size());
        std::vector<bool> topic_present(camera_topics_.size(), false);
        const int input_h = model_input_height_;
        const int input_w = model_input_width_;
        // For any camera that did not produce a frame this window we fill the batch slot with a
        // zero image at the engine's input size, so the model still sees a valid batch. The
        // corresponding `topic_present` entry stays false, so we never publish detections for
        // that slot — the fill image is purely a shape placeholder.
        for (size_t i = 0; i < camera_topics_.size(); ++i) {
            const std::string& topic = camera_topics_[i];
            auto it = processing_buffer.find(topic);
            if (it == processing_buffer.end()) {
                images.emplace_back(cv::Mat::zeros(cv::Size(input_w, input_h), CV_8UC3));
                continue;
            }
            topic_present[i] = true;
            try {
                auto cv_ptr = cv_bridge::toCvShare(it->second, sensor_msgs::image_encodings::RGB8);
                cv::Mat img;
                cv::cvtColor(cv_ptr->image, img, cv::COLOR_RGB2BGR);
                images.push_back(std::move(img));
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(),
                             "Failed to convert ROS image on topic %s: %s",
                             topic.c_str(), e.what());
                topic_present[i] = false;
                images.emplace_back(cv::Mat::zeros(cv::Size(input_w, input_h), CV_8UC3));
            }
        }

        std::vector<std::vector<Object>> objects = yoloV8_.detectObjects(images);

        // Per-batch detection log lives at INFO so it is visible by default during a race; per
        // detection details go to DEBUG to keep the steady-state log volume manageable.
        for (size_t i = 0; i < objects.size(); ++i) {
            if (!topic_present[i] || objects[i].empty()) {
                continue;
            }
            RCLCPP_INFO(this->get_logger(), "Detected %zu object(s) on %s",
                        objects[i].size(), camera_topics_[i].c_str());
            for (const auto& object : objects[i]) {
                RCLCPP_DEBUG(this->get_logger(), "  %s: %.3f",
                             yoloV8_.getClassName(object.label).c_str(), object.probability);
            }
        }

        publishDetections(images, objects, topic_present, processing_buffer);
    }

    void publishDetections(std::vector<cv::Mat>& images,
                           std::vector<std::vector<Object>>& objects,
                           const std::vector<bool>& topic_present,
                           const ImageBuffer& processing_buffer) {
        for (size_t i = 0; i < camera_topics_.size(); ++i) {
            if (!topic_present[i]) {
                continue;
            }
            const std::string& topic = camera_topics_[i];
            const auto image_msg = processing_buffer.at(topic);

            yolov8_interfaces::msg::Yolov8Detections detectionMsg;
            detectionMsg.header = image_msg->header;

            if (enable_one_channel_mask_) {
                publishOneChannelMask(objects[i], image_msg, detectionMsg, topic);
            }
            if (visualize_masks_) {
                visualizeMask(objects[i], images[i], topic, image_msg);
            }

            addObjectsToDetectionMsg(objects[i], detectionMsg);
            detection_publishers_[topic]->publish(detectionMsg);
        }
    }

    void visualizeMask(std::vector<Object>& objects, cv::Mat& image,
                       const std::string& topic,
                       const sensor_msgs::msg::Image::SharedPtr& image_msg) {
        yoloV8_.drawObjectLabels(image, objects);

        sensor_msgs::msg::Image displayImageMsg;
        cv_bridge::CvImage cv_image(image_msg->header, "bgr8", image);
        cv_image.toImageMsg(displayImageMsg);
        image_publishers_[topic]->publish(displayImageMsg);
    }

    void publishOneChannelMask(std::vector<Object>& objects,
                               const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                               yolov8_interfaces::msg::Yolov8Detections& detectionMsg,
                               const std::string& topic) {
        cv::Mat oneChannelMask;
        yoloV8_.getOneChannelSegmentationMask(objects, oneChannelMask,
                                              image_msg->height, image_msg->width);
        try {
            cv_bridge::CvImage cvBridgeOneChannelMask(image_msg->header, "mono8", oneChannelMask);
            detectionMsg.seg_mask_one_channel = *cvBridgeOneChannelMask.toImageMsg();
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception (one-channel mask): %s", e.what());
            return;
        }

        if (visualize_one_channel_mask_) {
            cv::Mat oneChannelMaskRGB8 = visualizeOneChannelMask(oneChannelMask);
            if (oneChannelMaskRGB8.empty()) {
                return;
            }
            try {
                cv_bridge::CvImage cvBridgeOneChannelMaskRGB8(image_msg->header, "rgb8", oneChannelMaskRGB8);
                one_channel_mask_publishers_[topic]->publish(*cvBridgeOneChannelMaskRGB8.toImageMsg());
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception (one-channel mask visualization): %s", e.what());
            }
        }
    }

    void addObjectsToDetectionMsg(const std::vector<Object>& objects,
                                  yolov8_interfaces::msg::Yolov8Detections& detectionMsg) const {
        // Start at 1: zero is reserved for "background" in the one-channel mask encoding.
        int index = 1;
        for (const auto& object : objects) {
            if (object.label + 1 > yoloV8_.getNumClasses()) {
                RCLCPP_ERROR(this->get_logger(),
                             "Label %d has no matching class name. Update CLASS_NAMES in yolov8.env.",
                             object.label);
                continue;
            }

            yolov8_interfaces::msg::Point2D point2DMsg;
            point2DMsg.x = object.rect.x;
            point2DMsg.y = object.rect.y;

            yolov8_interfaces::msg::Yolov8BBox bBoxMsg;
            bBoxMsg.top_left = point2DMsg;
            bBoxMsg.rect_width = object.rect.width;
            bBoxMsg.rect_height = object.rect.height;

            detectionMsg.indexes.push_back(index);
            detectionMsg.labels.push_back(object.label);
            detectionMsg.probabilities.push_back(object.probability);
            detectionMsg.class_names.push_back(yoloV8_.getClassName(object.label));
            detectionMsg.bounding_boxes.push_back(bBoxMsg);
            index++;
        }
    }

    cv::Mat visualizeOneChannelMask(const cv::Mat& oneChannelMask) const {
        try {
            cv::Mat oneChannelMask8U;
            oneChannelMask.convertTo(oneChannelMask8U, CV_8U);
            cv::Mat oneChannelMaskRGB8;
            cv::cvtColor(oneChannelMask8U, oneChannelMaskRGB8, cv::COLOR_GRAY2RGB);
            cv::normalize(oneChannelMaskRGB8, oneChannelMaskRGB8, 0, 255, cv::NORM_MINMAX);
            return oneChannelMaskRGB8;
        } catch (cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV exception in mask visualization: %s", e.what());
            return cv::Mat();
        }
    }

    std::vector<std::string> camera_topics_;
    std::string camera_topic_suffix_;
    double camera_buffer_hz_ = 30.0;
    bool visualize_masks_ = false;
    bool enable_one_channel_mask_ = false;
    bool visualize_one_channel_mask_ = false;

    // Cached from the model so the missing-topic fallback image matches the engine input.
    int model_input_height_ = 0;
    int model_input_width_ = 0;

    std::map<std::string, rclcpp::Publisher<yolov8_interfaces::msg::Yolov8Detections>::SharedPtr> detection_publishers_;
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> image_publishers_;
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> one_channel_mask_publishers_;
    rclcpp::TimerBase::SharedPtr buffer_timer_;
    ImageBuffer current_buffer_;
    std::mutex buffer_mutex_;

    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscriptions_;
    YoloV8& yoloV8_;
};

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;

    const std::string parseArgsError = parseArguments(argc, argv, config, onnxModelPath);
    if (!parseArgsError.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("yolov8"), "Argument parser: %s", parseArgsError.c_str());
        return 1;
    }

    RCLCPP_INFO(rclcpp::get_logger("yolov8"),
                "Creating YoloV8 engine -- this may take a while if the engine file is not cached.");
    YoloV8 yoloV8(onnxModelPath, config);
    RCLCPP_INFO(rclcpp::get_logger("yolov8"), "YoloV8 engine ready.");

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloV8Node>(yoloV8));
    rclcpp::shutdown();
    return 0;
}
