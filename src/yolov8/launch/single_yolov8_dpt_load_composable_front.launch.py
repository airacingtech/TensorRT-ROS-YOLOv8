'''
Generates the launch description for the yolov8_dpt package.
'''
from launch import LaunchDescription
from launch_ros.actions import Node, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os
from environs import Env
import yaml

def parse_camera_params(yaml_file):
    """
    Parse a camera calibration YAML file to extract F, CX, and CY parameters.
    
    Args:
        yaml_file (str): Path to the YAML file
        
    Returns:
        dict: Dictionary containing F, CX, and CY values
    """
    # Read the YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    
    # Extract camera matrix data
    camera_matrix = data['camera_matrix']['data']
    
    # In the camera matrix:
    # [0] is F_x (focal length x)
    # [4] is F_y (focal length y)
    # [2] is C_x (principal point x)
    # [5] is C_y (principal point y)
    
    # For simplicity, we'll use the average of F_x and F_y as F
    F_x = camera_matrix[0]  # 249.6918002633213
    F_y = camera_matrix[4]  # 250.10741579188723
    F = (F_x + F_y) / 2
    
    CX = camera_matrix[2]   # 260.4798634081818
    CY = camera_matrix[5]   # 193.1699690250926
    
    return F, CX, CY

def generate_launch_description():
    """
    Generates the launch description for the yolov8 package.

    Returns:
        LaunchDescription: The launch description object.
    """
    package_share_dir = get_package_share_directory('yolov8_dpt')
    iac_launch_package_share_dir = get_package_share_directory('iac_launch')
    models_dir = os.path.join(package_share_dir, 'models')
    env_file_path = os.path.join(package_share_dir, 'yolov8.env')
    # Get parameters from environment variables specifed in .env file
    # Check if .env file exists
    if not os.path.exists(env_file_path):
        raise FileNotFoundError("Please create a yolov8.env file in the root of the yolov8 \
                                workspace and rebuild the package so it can be installed in the \
                                package's share directory.")

    env = Env()
    env.read_env(env_file_path)

    #TODO: Make it such that it doesn't use the absolute path but relative
    yolo_onnx_path = env.str("YOLO_ONNX_PATH")
    dpt_engine_path = env.str("DPT_ENGINE_PATH")
    # camera_topics = env.str("CAMERA_TOPICS").split(",")
    camera_topics = ["/vimba_front"]
    camera_topic_suffix = env.str("CAMERA_TOPIC_SUFFIX")
    camera_buffer_hz = env.float("CAMERA_BUFFER_HZ")
    visualize_masks = env.bool("VISUALIZE_IMAGE_MASKS")
    visualize_image_pointcloud = env.bool("VISUALIZE_IMAGE_POINTCLOUD")
    publish_detection = env.bool("PUBLISH_DETECTION")
    nice_level = env.int("NICE_LEVEL")
    precision = env.str("PRECISION")
    calibration_data_directory = env.str("CALIBRATION_DATA_DIRECTORY")
    probability_threshold = env.float("PROBABILITY_THRESHOLD")
    nms_threshold = env.float("NMS_THRESHOLD")
    top_k = env.int("TOP_K")
    seg_channels = env.int("SEG_CHANNELS")
    seg_h = env.int("SEG_H")
    seg_w = env.int("SEG_W")
    segmentation_threshold = env.float("SEGMENTATION_THRESHOLD")
    target_width = env.int("TARGET_WIDTH")
    target_height = env.int("TARGET_HEIGHT")
    # class_names = env("CLASS_NAMES").split(",")

    # Dynamically Getting Camera Calibration parameters for single camera
    camera_facing = camera_topics[0][7:] # Getting First Camera Topic
    camera_calibration_filepath = os.path.join(iac_launch_package_share_dir, "param", "cameras_param", f"cam_{camera_facing}_calib.yaml")
    # f"/param/cameras_param/cam_{camera_facing}_calib.yaml"
    if os.path.exists(camera_calibration_filepath):
        print(f"Camera calibration file found: {camera_calibration_filepath}")
        focal_length, cx, cy = parse_camera_params(camera_calibration_filepath)
    else: # default values
        print(f"Camera calibration file not found at: {camera_calibration_filepath}. Using default values.")
        focal_length = 256.6171
        cx = 254.2069
        cy = 195.0000

    print("YOLOv8 Parameters:")
    print(f"yolo_onnx_path: {yolo_onnx_path}")
    print(f"dpt_engine_path: {dpt_engine_path}")
    print(f"focal_length: {focal_length}") # TODO: dpt current implementation assumes single camera
    print(f"cx: {cx}") # TODO: dpt current implementation assumes single camera
    print(f"cy: {cy}") # TODO: dpt current implementation assumes single camera
    print(f"camera_topics: {camera_topics}")
    print(f"camera_topic_suffix: {camera_topic_suffix}")
    print(f"camera_buffer_hz: {camera_buffer_hz}")
    print(f"visualize_masks: {visualize_masks}")
    print(f"visualize_image_pointcloud: {visualize_image_pointcloud}")
    print(f"publish_detection: {publish_detection}")
    print(f"nice_level: {nice_level}")
    print(f"precision: {precision}")
    print(f"calibration_data_directory: {calibration_data_directory}")
    print(f"probability_threshold: {probability_threshold}")
    print(f"nms_threshold: {nms_threshold}")
    print(f"top_k: {top_k}")
    print(f"seg_channels: {seg_channels}")
    print(f"seg_h: {seg_h}")
    print(f"seg_w: {seg_w}")
    print(f"segmentation_threshold: {segmentation_threshold}")
    print(f"target_width: {target_width}")
    print(f"target_height: {target_height}")
    
    # print(f"class_names: {class_names}")

    # Convert the list of class names into several strings separated by "" to be read as a command
    # line argument
    # class_names = ' '.join([f'"{class_name}"' for class_name in class_names])

    # TODO Pull parameters out of ros_segmentation and place them here
    container_name = f"{camera_facing}_camera_camera_container"
    return LaunchDescription([
        LoadComposableNodes(
            target_container=container_name,
            composable_node_descriptions=[
                ComposableNode(
                    package='yolov8_dpt',
                    plugin='yolov8_dpt::YoloV8Node',
                    name='yolov8_dpt_node',
                    parameters=[{
                        'yolo_onnx_path': yolo_onnx_path,
                        'dpt_engine_path': dpt_engine_path,
                        'focal_length': focal_length,
                        'cx': cx,
                        'cy': cy,
                        'camera_topics': camera_topics,
                        'camera_topic_suffix': camera_topic_suffix,
                        'camera_buffer_hz': camera_buffer_hz,
                        'visualize_masks': visualize_masks,
                        'visualize_image_pointcloud': visualize_image_pointcloud,
                        'publish_detection': publish_detection,
                        'target_width': target_width,
                        'target_height': target_height,
                    }],
                )
            ]
        )
    ])
