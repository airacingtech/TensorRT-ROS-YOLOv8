# Segmentation masks and bounding boxes for a single frame.

std_msgs/Header header

# 1-based instance ids (0 is reserved for background in seg_mask_one_channel).
int32[] indexes

# Class label and probability for each detection (parallel to indexes).
int32[] labels
float64[] probabilities
string[] class_names

# Single-channel mask combining all instances. Each pixel is 0 for background or the
# 1-based index of the instance that owns it. Encoded as mono8.
sensor_msgs/Image seg_mask_one_channel

# Axis-aligned bounding box for each instance (parallel to indexes).
yolov8_interfaces/Yolov8BBox[] bounding_boxes
