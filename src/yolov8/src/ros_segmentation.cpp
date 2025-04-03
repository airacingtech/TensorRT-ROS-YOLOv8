# include <mutex>

#include "yolov8.h"
#include "dpt.h"
#include "cmd_line_util.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "cv_bridge/cv_bridge.hpp"

// Custom ROS2 message types where the names of the hpp files are snake_case
#include "yolov8_interfaces/msg/point2_d.hpp"
#include "yolov8_interfaces/msg/yolov8_detections.hpp"
#include "yolov8_interfaces/msg/yolov8_seg_mask.hpp"
#include "yolov8_interfaces/msg/yolov8_b_box.hpp"

#include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_DEBUG_THROTTLED(rclcpp::get_logger("rclcpp"), 100, "Debug message");
using std::placeholders::_1;
#define DPT_UPWARD_SHIFT 0 // TODO: Need more thinking, probably this needs to be done in the dataset of the depth model training.
LoggerTRT logger_trt;

namespace yolov8_dpt
{
    class YoloV8Node : public rclcpp::Node
    {
        public:
            YoloV8Node(const rclcpp::NodeOptions & options)
            : Node("yolo_v8_dpt", options)
            {
                // If you want to do ros2 run, change the parameters here:
                // Declare parameters with their default values.
                this->declare_parameter<std::string>("yolo_onnx_path", "/home/autera-admin/ART/race_common/src/external/TensorRT-ROS-YOLOv8/src/yolov8/models/best.onnx");
                this->declare_parameter<std::string>("dpt_engine_path","/home/autera-admin/ART/race_common/src/external/TensorRT-ROS-YOLOv8/src/yolov8/models/engines/DepthAnythingv1_11-Dec_12-04-50740ac911a2_latest_opset19.engine");
                this->declare_parameter<float>("focal_length", 256.6171); // # Focal length in x
                this->declare_parameter<float>("cx", 254.2069); // # Principal point x
                this->declare_parameter<float>("cy", 195.0000); // 202.1108  # Principal point y
                this->declare_parameter<std::vector<std::string>>("camera_topics", std::vector<std::string>{"/vimba_rear"});
                this->declare_parameter<std::string>("camera_topic_suffix", "/image/ptr");
                this->declare_parameter<float>("camera_buffer_hz", 25.0);
                this->declare_parameter<bool>("visualize_masks", true);
                this->declare_parameter<bool>("visualize_image_pointcloud", true);
                this->declare_parameter<bool>("publish_detection", true);
                this->declare_parameter<int>("target_width", 1056);
                this->declare_parameter<int>("target_height", 1056);

                // Retrieve parameters and store them in member variables.
                this->get_parameter("yolo_onnx_path", yolo_onnx_path_);
                this->get_parameter("dpt_engine_path", dpt_engine_path_);
                this->get_parameter("focal_length", F);
                this->get_parameter("cx", CX);
                this->get_parameter("cy", CY);
                this->get_parameter("camera_topics", camera_topics_);
                this->get_parameter("camera_topic_suffix", camera_topic_suffix_);
                this->get_parameter("camera_buffer_hz", camera_buffer_hz_);
                this->get_parameter("visualize_masks", visualize_masks_);
                this->get_parameter("visualize_image_pointcloud", visualize_image_pointcloud_);
                this->get_parameter("publish_detection", publish_detection_);
                this->get_parameter("target_width", target_width_);
                this->get_parameter("target_height", target_height_);

                // Creating the models here
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Creating YoloV8 engine --- Could take a while if Engine file is not already built and cached.");
                yoloV8_ = std::make_unique<YoloV8>(yolo_onnx_path_, yolov8_config_);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "YoloV8 engine created and loaded into memory.");
                
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Creating DepthAnything engine --- Could take a while if Engine file is not already built and cached.");
                dpt_ = std::make_unique<DepthAnything>();
                dpt_->init(dpt_engine_path_, logger_trt);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "DepthAnything engine created and loaded into memory.");

                // Create timer for camera synchronization
                float buffer_duration = 1 / camera_buffer_hz_;
                buffer_timer_ = rclcpp::create_wall_timer(
                    std::chrono::duration<float>(buffer_duration),
                    std::bind(&YoloV8Node::batchBufferCallback, this),
                    nullptr,
                    this->get_node_base_interface().get(),
                    this->get_node_timers_interface().get()
                );

                // Print Camera Topics
                RCLCPP_INFO(this->get_logger(), "Camera Topics:");
                for (const std::string& topic : camera_topics_) {
                    RCLCPP_INFO(this->get_logger(), "  %s", topic.c_str());
                }

                // Check if model batch size matches the number of camera topics
                if (camera_topics_.size() != yoloV8_->getBatchSize()) {
                    throw std::runtime_error("Model batch size (" + std::to_string(yoloV8_->getBatchSize()) +
                        ") does not match the number of camera topics (" + std::to_string(camera_topics_.size()) + ")");
                }

                // Create subscribers and publishers for all cameras
                rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 10));
                qos_profile.best_effort();
                qos_profile.durability_volatile();
                rclcpp::SubscriptionOptions sub_options;
                sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
                
                for (const std::string& topic : camera_topics_) {
                    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
                        topic + camera_topic_suffix_, qos_profile,
                        [this, topic](const sensor_msgs::msg::Image::SharedPtr msg)
                        {
                            this->addToBufferCallback(msg, topic);
                        },
                        sub_options
                    );
                    RCLCPP_INFO(this->get_logger(), "Subscribed to topic: %s", (topic + camera_topic_suffix_).c_str());
                    
                    subscriptions_.push_back(subscription_);

                    // Chris: Publishing detections should be a thing of the past
                    // detection_publishers_[topic] = this->create_publisher<yolov8_interfaces::msg::Yolov8Detections>(
                    //     "/yolov8" + topic + "/detections", qos_profile
                    // );

                    if (visualize_masks_) {
                        image_publishers_[topic] = this->create_publisher<sensor_msgs::msg::Image>(
                            "/yolov8" + topic + "/image", qos_profile
                        );
                        one_channel_mask_publishers_[topic] = this->create_publisher<sensor_msgs::msg::Image>(
                            "/yolov8" + topic + "/seg_mask_one_channel", qos_profile
                        );
                        detection_pointcloud_publishers_[topic] = this->create_publisher<sensor_msgs::msg::Image>(
                            "/dpt" + topic + "/depth_image", qos_profile
                        );
                    }

                    if (visualize_image_pointcloud_) {
                        full_point_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                            "/dpt/full_point_cloud", qos_profile
                        );
                    }
                    
                    if (publish_detection_) {

                        filtered_point_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                            "/dpt/filtered_point_cloud", qos_profile
                        );
                    }


                }
            }

        private:
            /*
            * Add the image message from the given image topic to the camera buffer. Note the preprocessing not done here
            * since a given image might be replaced by a newer image before the camera buffer is batched to the
            * neural network.
            *
            * @param image_msg: The image message from the camera
            * @param topic: The ROS topic the image message was published on
            */
            void addToBufferCallback(const sensor_msgs::msg::Image::SharedPtr &image_msg, const std::string topic) {
                // For verifying Composable node Addresses
                std::stringstream ss;
                ss << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(image_msg.get());
                std::cout << "Received image message on topic " << topic << " on address: " << ss.str() << std::endl;
                std::unique_lock<std::mutex> lock(buffer_mutex_);
                current_buffer_[topic] = image_msg; // No need to move since RMW will keep the message valid

                // Check if ready for batching
                if (current_buffer_.size() == camera_topics_.size()) {
                    lock.unlock();
                    batchBufferCallback();
                }
            }

            void batchBufferCallback() {
                // Swap the current buffer with the processing buffer
                std::unique_lock<std::mutex> lock(buffer_mutex_);
                std::swap(current_buffer_, processing_buffer_);
                // Clear the current buffer
                current_buffer_.clear(); // Allows RMW to deallocate image pointers
                lock.unlock();

                // Reset buffer timer
                buffer_timer_->reset();

                std::map<std::string, cv::Mat> images_map;

                // Check what camera topics are in the processing buffer
                auto [processing_topics, missing_topics] = checkCameraTopicsInBuffer();
                if (missing_topics.size() == camera_topics_.size()) {
                    std::cout << "No camera topics in the processing buffer at time " << this->now().seconds() << std::endl;
                    return;
                } else {
                    // Add black images for missing topics
                    // Log target width and height
                    RCLCPP_INFO(this->get_logger(), "Target image size: %dx%d", target_width_, target_height_);
                    for (const std::string& topic : missing_topics) {
                        // TODO: MAKE IMAGE SIZE DYNAMIC
                        images_map[topic] = cv::Mat::zeros(cv::Size(target_width_, target_height_), CV_8UC3);
                    }
                }

                // Preprocess the input
                preprocess_callback(images_map);
                // Check the image sizes
                for (const auto& pair : images_map) {
                    RCLCPP_INFO(this->get_logger(), "Preprocessed image size: %dx%d", pair.second.cols, pair.second.rows);
                }

                // Place map values into a vector
                std::vector<cv::Mat> images;
                for (const auto& pair : images_map) {
                    images.push_back(pair.second);
                }

                // Run inference
                RCLCPP_INFO(this->get_logger(), "Running YoloV8 inference");
                std::vector<std::vector<Object>> objects = yoloV8_->detectObjects(images);
                RCLCPP_INFO(this->get_logger(), "Inference YoloV8 complete");

                // For REAR Perception, this only:
                RCLCPP_INFO(this->get_logger(), "Running DPT inference");
                cv::Mat depth_output = dpt_->predict(images[0], (!visualize_masks_ && !publish_detection_), DPT_UPWARD_SHIFT); // Zero copy mode if images not needed downstream
                RCLCPP_INFO(this->get_logger(), "Inference DPT complete");
                if (visualize_image_pointcloud_) {
                    publishFullPointCloud(depth_output, camera_topics_[0]);
                }
                RCLCPP_INFO(this->get_logger(), "Depth Output Published");


                if (objects.size() == 0) {
                std::cout << "No objects detected at time " << this->now().seconds() << std::endl;
                }

                size_t i = 0;
                for (const auto& batch : objects) {
                    if (!batch.empty()) {
                        std::string topic = camera_topics_[i];
                        RCLCPP_INFO(this->get_logger(), "Detected %zu object(s) on %s", batch.size(), topic.c_str());
                        for (const auto& object : batch) {
                            std::cout << "\tDetected : " << yoloV8_->getClassName(object.label) << ", Prob: " << object.probability << std::endl;
                            RCLCPP_INFO(this->get_logger(), "\t%s: %f", yoloV8_->getClassName(object.label).c_str(), object.probability);
                        }
                    }
                    i++;
                }
                // RCLCPP_INFO(this->get_logger(), "Detected %zu objects across all cameras", total_objects);
                // RCLCPP_INFO(this->get_logger(), "Typeid: %s", typeid(objects[0].boxMask).name());
                // RCLCPP_INFO(this->get_logger(), "Mask Shape: %s", objects[0].boxMask.size());

                // Postprocess the output and publish the results
                postprocess_callback(images_map, objects, images, depth_output, missing_topics);
            }

            /*
            * Check number of camera topics in the processing buffer and return any missing topics
            *
            * @return A pair of vectors containing the camera topics in the processing buffer and the missing topics
            */
            std::pair<std::vector<std::string>, std::vector<std::string>> checkCameraTopicsInBuffer() {
                std::vector<std::string> missing_topics;
                std::vector<std::string> processing_topics;
                RCLCPP_INFO(this->get_logger(), "Checking camera topics in the processing buffer");
                RCLCPP_INFO(this->get_logger(), "Processing buffer size: %zu", processing_buffer_.size());
                RCLCPP_INFO(this->get_logger(), "Camera topics size: %zu", camera_topics_.size());
                if (processing_buffer_.size() == camera_topics_.size()) {
                    std::cout << "All camera topics are in the processing buffer" << std::endl;
                    for (const std::string& topic : camera_topics_) {
                        processing_topics.push_back(topic);
                    }
                } else {
                    if (processing_buffer_.size() == 0) {
                        std::cout << "No camera topics are in the processing buffer" << std::endl;
                    }
                    for (const std::string& topic : camera_topics_) {
                        if (processing_buffer_.find(topic) == processing_buffer_.end()) {
                            std::cout << "Camera topic " << topic << " is missing from the processing buffer" << std::endl;
                            missing_topics.push_back(topic);
                        } else {
                            processing_topics.push_back(topic);
                        }
                    }
                }

                return std::make_pair(processing_topics, missing_topics);
            }

            void preprocess_callback(std::map<std::string, cv::Mat>& images) {
                for (const auto& pair : processing_buffer_) {
                    std::string camera_topic = pair.first;
                    const sensor_msgs::msg::Image::SharedPtr image_msg = pair.second;
                    try
                        {
                            // Share the memory with the original image
                            // TODO: Should this be a copy to deal with a cleared buffer?
                            cv_bridge::CvImageConstPtr cv_ptr;
                            cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::RGB8);

                            // Convert from RGB8 to BGR8
                            cv::Mat img = cv_ptr->image;
                            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
                            images[camera_topic] = img;
                        } catch (cv_bridge::Exception& e) {
                            RCLCPP_ERROR(this->get_logger(), "Failed to convert ROS image message on topic %s \
                                due to cv_bridge error: %s", camera_topic.c_str(), e.what());
                            continue;
                        }
                }
            }

            /*
            * Convert output(s) from Neural Network to ROS messages and publish them
            *
            * @param objects: Batch of detected objects from the neural network
            * @param images: The images that the objects were detected in
            * @param missing_topics: The camera topics that were missing from the processing buffer
            */
            void postprocess_callback(std::map<std::string, cv::Mat> images_map, std::vector<std::vector<Object>> objects,
                    std::vector<cv::Mat> images, 
                    cv::Mat& depth_map,
                    std::vector<std::string> missing_topics) {

                int i = 0;
                for (const auto& pair : images_map) {
                    const std::string topic = pair.first;
                    // Skip any missing camera topics since they will have a black image
                    if (std::find(missing_topics.begin(), missing_topics.end(), topic) != missing_topics.end()) {
                        i++;
                        std::cout << "Skipping missing topic " << topic << std::endl;
                        continue;
                    }

                    const sensor_msgs::msg::Image::SharedPtr image_msg = processing_buffer_[topic];
                    cv::Mat image = images[i];
                    // ROS message to publish the detections
                    yolov8_interfaces::msg::Yolov8Detections detectionMsg;
                    detectionMsg.header = image_msg->header;

                    if (publish_detection_) {
                        // TODO fix how batched objects are handled
                        publishOneChannelMaskAndPointCloudDetection(
                            objects[i], 
                            image_msg, 
                            detectionMsg,
                            depth_map,
                            topic
                        );
                    }

                    // Draw the segmentation masks and bounding boxes on the image
                    // to visualize the detections
                    if (visualize_masks_) {
                        // TODO fix how batched objects are handled
                        visualizeMask(objects[i], image, topic, image_msg);
                    }

                    // Chris: Detection Publishing should be something of the past
                    // // Convert detected objects to ROS message
                    // // TODO fix how batched objects are handled
                    // addObjectsToDetectionMsg(objects[i], detectionMsg);

                    // // Publish the detections
                    // detection_publishers_[topic]->publish(detectionMsg);
                    i++;
                }
            }

            // Function to publish a individual object point cloud.
            // Only pixels that are nonzero in the object_mask (CV_8UC1)
            // and have valid depth (depth > 0) are back-projected.
            void publishDetectionPointCloud(const cv::Mat &depth_map, const cv::Mat &object_mask, std::string camera_topic)
            {
                if (depth_map.empty() || object_mask.empty()) {
                    RCLCPP_WARN(rclcpp::get_logger("ObjectPointCloudPublisher"), "Empty depth map or object mask provided");
                    return;
                }
                if (depth_map.type() != CV_32FC1) {
                    RCLCPP_ERROR(rclcpp::get_logger("ObjectPointCloudPublisher"), "Depth map must be of type CV_32FC1: Depth map type: %d", depth_map.type());
                    return;
                }
                if (object_mask.type() != CV_8UC1) {
                    RCLCPP_ERROR(rclcpp::get_logger("ObjectPointCloudPublisher"), "Object mask must be of type CV_8UC1");
                    return;
                }
                if (depth_map.size() != object_mask.size()) {
                    RCLCPP_ERROR(rclcpp::get_logger("ObjectPointCloudPublisher"), "Depth map and object mask must have the same dimensions");
                    return;
                }

                int width = depth_map.cols;
                int height = depth_map.rows;
                std::vector<float> points;

                // Loop through each pixel. Only process pixels where the mask indicates the object.
                for (int v = 0; v < height; ++v) {
                    for (int u = 0; u < width; ++u) {
                        // Check if the current pixel belongs to the object.
                        if (object_mask.at<uchar>(v, u) == 0)
                            continue;

                        float d = depth_map.at<float>(v, u);
                        if (d <= 0.0f) continue;  // Skip pixels with no depth
                        float z = -((v - CY) / F) * d;
                        float y = -((u - CX) / F) * d;
                        float x = d;
                        points.push_back(x);
                        points.push_back(y);
                        points.push_back(z);
                    }
                }

                // Build the PointCloud2 message for 
                sensor_msgs::msg::PointCloud2 cloud_msg;
                cloud_msg.header.stamp = rclcpp::Clock().now();
                cloud_msg.header.frame_id = getFrameIdFromTopic(camera_topic);
                cloud_msg.height = 1;
                cloud_msg.width = static_cast<uint32_t>(points.size() / 3);
                cloud_msg.is_dense = false;
                cloud_msg.is_bigendian = false;
                cloud_msg.point_step = 3 * sizeof(float);
                cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

                sensor_msgs::msg::PointField field;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                
                field.name = "x";
                field.offset = 0;
                cloud_msg.fields.push_back(field);
                
                field.name = "y";
                field.offset = 4;
                cloud_msg.fields.push_back(field);
                
                field.name = "z";
                field.offset = 8;
                cloud_msg.fields.push_back(field);

                size_t data_size = points.size() * sizeof(float);
                cloud_msg.data.resize(data_size);
                memcpy(cloud_msg.data.data(), points.data(), data_size);

                filtered_point_cloud_publisher_->publish(cloud_msg);
            }

            // Function to publish the full depth map as a point cloud.
            // Skips pixels where the depth value is 0.
            void publishFullPointCloud(const cv::Mat &depth_map, const std::string &camera_topic) {
                if (depth_map.empty()) {
                    RCLCPP_WARN(rclcpp::get_logger("DepthPointCloudPublisher"), "Empty depth map provided");
                    return;
                }
                if (depth_map.type() != CV_32FC1) {
                    RCLCPP_ERROR(rclcpp::get_logger("ObjectPointCloudPublisher"), "Depth map must be of type CV_32FC1: Depth map type: %d", depth_map.type());
                    return;
                }
            
                int width = depth_map.cols;
                int height = depth_map.rows;
                std::vector<float> points;  // Each point has x, y, z
            
                // Loop through each pixel and back-project if depth > 0.
                for (int v = 0; v < height; ++v) {
                    for (int u = 0; u < width; ++u) {
                        float d = depth_map.at<float>(v, u);
                        if (d <= 0.0f) continue;  // Skip pixels with no depth
            
                        // Back-project pixel (u, v) into 3D space.
                        float z = -((v - CY) / F) * d;
                        float y = -((u - CX) / F) * d;
                        float x = d;
                        points.push_back(x);
                        points.push_back(y);
                        points.push_back(z);
                    }
                }
            
                // Build the PointCloud2 message.
                sensor_msgs::msg::PointCloud2 cloud_msg;
                cloud_msg.header.stamp = rclcpp::Clock().now();
                cloud_msg.header.frame_id = getFrameIdFromTopic(camera_topic);
                cloud_msg.height = 1;
                cloud_msg.width = static_cast<uint32_t>(points.size() / 3);
                cloud_msg.is_dense = false;
                cloud_msg.is_bigendian = false;
                cloud_msg.point_step = 3 * sizeof(float);
                cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
            
                // Define fields for x, y, and z.
                sensor_msgs::msg::PointField field;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                
                field.name = "x";
                field.offset = 0;
                cloud_msg.fields.push_back(field);
                
                field.name = "y";
                field.offset = 4;
                cloud_msg.fields.push_back(field);
                
                field.name = "z";
                field.offset = 8;
                cloud_msg.fields.push_back(field);
            
                // Copy the point data into the message.
                size_t data_size = points.size() * sizeof(float);
                cloud_msg.data.resize(data_size);
                memcpy(cloud_msg.data.data(), points.data(), data_size);
            
                // Publish the point cloud.
                full_point_cloud_publisher_->publish(cloud_msg);
            }

            /*
            * Visualize the segmentation masks and bounding boxes on the image and publish it
            * to the given topic.
            *
            * @param objects: Detected objects from the neural network
            * @param image: The image to visualize the masks on
            * @param topic: The ROS topic to publish the image to
            * @param image_msg: The original ROS image message from the camera
            */
            void visualizeMask(std::vector<Object> objects, cv::Mat image,
                    const std::string topic,
                    const sensor_msgs::msg::Image::SharedPtr image_msg) {
                // Draw the object labels on the image
                yoloV8_->drawObjectLabels(image, objects);

                // Turn cv_image into sensor_msgs::msg::Image
                sensor_msgs::msg::Image displayImageMsg;
                // TODO: Change to rgb8?
                cv_bridge::CvImagePtr cv_image = std::make_shared<cv_bridge::CvImage>(
                    image_msg->header, "bgr8", image
                );
                cv_image->toImageMsg(displayImageMsg);
                // Publish segmented image as ROS message
                image_publishers_[topic]->publish(displayImageMsg);
            }

            /*
            * Create a one channel mask for all segmentation objects and publish it. Adds oneChannelMask
            * to detectionMsg. Optionally visualize the mask.
            *
            * @param objects: Detected objects from the neural network
            * @param visualize_one_channel_mask: Whether to visualize the one channel mask
            * @param image_msg: Original ROS image from the camera
            * @param detectionMsg: ROS message to publish detections
            * @param topic: The ROS topic to publish the mask to
            */
            void publishOneChannelMaskAndPointCloudDetection(std::vector<Object> objects,
                    const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                    yolov8_interfaces::msg::Yolov8Detections &detectionMsg,
                    cv::Mat& depthimage,
                    const std::string topic) {
                cv::Mat oneChannelMask;
                int img_width = image_msg->width;
                int img_height = image_msg->height;
                yoloV8_->getOneChannelSegmentationMask(objects, oneChannelMask, img_height, img_width, 0.9);
                publishDetectionPointCloud(depthimage, oneChannelMask, topic);

                // Publish the one channel mask with RGB color for visualization only
                if (visualize_masks_) {

                    // Use ROS cv_bridge to convert cv::Mat to sensor_msgs::msg::Image and take header from original camera image
                    try {
                        cv_bridge::CvImage cvBridgeOneChannelMask = cv_bridge::CvImage(
                            image_msg->header, "mono8", oneChannelMask
                        );
                        detectionMsg.seg_mask_one_channel = *cvBridgeOneChannelMask.toImageMsg();
                    } catch (cv_bridge::Exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                        return;
                    }

                    // Publish the overlayed pointcloud
                    cv::Mat oneChannelMaskRGB8 = visualizeOneChannelMask(oneChannelMask);
                    cv::Mat oneChannelDepthRGB8 = visualizeDepth(depthimage);

                    // Publish the one channel mask with RGB color for visualization only
                    try {
                        cv_bridge::CvImage cvBridgeOneChannelMaskRGB8 = cv_bridge::CvImage(
                            image_msg->header, "rgb8", oneChannelMaskRGB8
                        );
                        one_channel_mask_publishers_[topic]->publish(*cvBridgeOneChannelMaskRGB8.toImageMsg());
                        
                    } catch (cv_bridge::Exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                        return;
                    }

                    // Publish the depth with RGB color for visualization only
                    try {
                        cv_bridge::CvImage cvBridgeOneChannelDepthRGB8 = cv_bridge::CvImage(
                            image_msg->header, "rgb8", oneChannelDepthRGB8
                        );
                        detection_pointcloud_publishers_[topic]->publish(*cvBridgeOneChannelDepthRGB8.toImageMsg());
                        
                    } catch (cv_bridge::Exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                        return;
                    }
                }
            }

            /*
            * Add detected objects to the yolov8detections message
            *
            * @param objects: Detected objects from the neural network
            * @param detectionMsg: The yolov8detections message to add the objects to
            */
            void addObjectsToDetectionMsg(std::vector<Object> objects,
                    yolov8_interfaces::msg::Yolov8Detections& detectionMsg) const {
                // Convert detected objects to ROS message
                // Start at index 1 because index 0 is the background class
                int index = 1;
                for (auto & object : objects) {
                    int label = object.label;
                    float prob = object.probability;

                    // Create yolov8_obj bounding box message
                    yolov8_interfaces::msg::Yolov8BBox bBoxMsg;
                    yolov8_interfaces::msg::Point2D point2DMsg;
                    point2DMsg.x = object.rect.x;
                    point2DMsg.y = object.rect.y;
                    bBoxMsg.top_left = point2DMsg;
                    bBoxMsg.rect_width = object.rect.width;
                    bBoxMsg.rect_height = object.rect.height;

                    // Add segmentation masks, bounding boxes, and class info to yolov8detections message
                    detectionMsg.indexes.push_back(index);
                    detectionMsg.labels.push_back(label);
                    detectionMsg.probabilities.push_back(prob);
                    if (label + 1 > yoloV8_->getNumClasses()) {
                        RCLCPP_ERROR(this->get_logger(), "Label %d does not have a corresponding class name. Did you update yolov8.env to include all classes?", label);
                        continue;
                    }
                    detectionMsg.class_names.push_back(yoloV8_->getClassName(label));
                    detectionMsg.bounding_boxes.push_back(bBoxMsg);

                    index++;
                }
            }

            /*
            * Create visualization of the one channel mask as an RGB image
            *
            * @param oneChannelMask: The one channel mask to visualize
            * @return The one channel mask as a RGB image
            */
            cv::Mat visualizeOneChannelMask(cv::Mat oneChannelMask) const {
                try {
                    // Draw the one channel mask on the image
                    // Convert the one channel mask to 8-bit
                    cv::Mat oneChannelMask8U;
                    oneChannelMask.convertTo(oneChannelMask8U, CV_8U);
                    // Convert the one channel mask to 3-channel RGB
                    cv::Mat oneChannelMaskRGB8;
                    cv::cvtColor(oneChannelMask8U, oneChannelMaskRGB8, cv::COLOR_GRAY2RGB);
                    // Normalize the one channel mask to 0-255 so it can be displayed as an RGB image
                    cv::normalize(oneChannelMaskRGB8, oneChannelMaskRGB8, 0, 255, cv::NORM_MINMAX);
                    return oneChannelMaskRGB8;
                } catch (cv::Exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "OpenCV exception: %s", e.what());
                    return cv::Mat();
                }
            }

            cv::Mat visualizeDepth(cv::Mat& depth_mat) const {
                try {
                    cv::normalize(depth_mat, depth_mat, 0, 255, cv::NORM_MINMAX, CV_8U);
                    // Create a colormap from the depth data
                    cv::Mat colormap;
                    cv::applyColorMap(depth_mat, colormap, cv::COLORMAP_INFERNO);
                    return colormap;
                } catch (cv::Exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "OpenCV exception: %s", e.what());
                    return cv::Mat();
                }
            }

            std::vector<std::string> camera_topics_;
            std::string yolo_onnx_path_;
            std::string dpt_engine_path_;
            std::string camera_topic_suffix_;
            double F;
            double CX;
            double CY;

            float camera_buffer_hz_ = 25;
            bool visualize_masks_;
            bool visualize_image_pointcloud_;
            bool publish_detection_;
            int target_width_;
            int target_height_;
            double target_aspect;

            // std::map<std::string, rclcpp::Publisher<yolov8_interfaces::msg::Yolov8Detections>::SharedPtr> detection_publishers_;
            std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> image_publishers_;
            std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> one_channel_mask_publishers_;
            std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> detection_pointcloud_publishers_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr full_point_cloud_publisher_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_point_cloud_publisher_;

            rclcpp::TimerBase::SharedPtr buffer_timer_;
            std::map<std::string, sensor_msgs::msg::Image::SharedPtr> current_buffer_;
            std::map<std::string, sensor_msgs::msg::Image::SharedPtr> processing_buffer_;
            std::mutex buffer_mutex_;

            std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscriptions_;
            std::unique_ptr<YoloV8> yoloV8_;
            std::unique_ptr<DepthAnything> dpt_;
            YoloV8Config yolov8_config_;
    };
}

// Register the component
RCLCPP_COMPONENTS_REGISTER_NODE(yolov8_dpt::YoloV8Node)

// Only include the main function when building the executable, not the component library
#if defined(BUILD_EXECUTABLE) || !defined(BUILD_COMPONENT)
int main(int argc, char *argv[]) {
    // Create ROS2 Node
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    rclcpp::spin(std::make_shared<yolov8_dpt::YoloV8Node>(options));
    rclcpp::shutdown();
    return 0;
}
#endif

// int main(int argc, char *argv[]) {

//     // Create ROS2 Node
//     rclcpp::init(argc, argv);
//     rclcpp::NodeOptions options;
//     rclcpp::spin(std::make_shared<yolov8_dpt::YoloV8Node>(options));
//     rclcpp::shutdown();
//     return 0;
// }

// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(yolov8_dpt::YoloV8Node)