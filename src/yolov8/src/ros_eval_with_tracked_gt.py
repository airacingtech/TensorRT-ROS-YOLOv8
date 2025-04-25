#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2
import numpy as np
import csv
import os
from datetime import datetime
import math
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from collections import deque

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('pointcloud_processor')
        
        # Create separate callback groups for subscribers and timers
        sub_callback_group = MutuallyExclusiveCallbackGroup()
        processing_timer_callback_group = MutuallyExclusiveCallbackGroup()
        csv_timer_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Declare parameters
        self.declare_parameter('groundTruth_topic', '/tracked_objects')
        self.declare_parameter('camera_topic', '/dpt/filtered_point_cloud')
        self.declare_parameter('processing_frequency', 15.0)  # Hz
        self.declare_parameter('csv_write_frequency', 5.0)    # Hz
        self.declare_parameter('csv_output_path', 'detection_errors.csv')
        
        # Processing parameters
        self.declare_parameter('camera_agg_mode', 'MEDIAN')  # MEDIAN or AVERAGE
        self.declare_parameter('camera_temporal', True)     # Whether to use temporal aggregation
        self.declare_parameter('temporal_buffer_size', 5)    # Size of temporal buffer
        self.declare_parameter('eval_radius', 1.0) # Radius surrounding the groundTruth point
        
        # Get parameters
        groundTruth_topic = self.get_parameter('groundTruth_topic').value
        camera_topic = self.get_parameter('camera_topic').value
        processing_frequency = self.get_parameter('processing_frequency').value
        csv_write_frequency = self.get_parameter('csv_write_frequency').value
        self.csv_output_path = self.get_parameter('csv_output_path').value
        
        # Get processing parameters
        self.camera_agg_mode = self.get_parameter('camera_agg_mode').value
        self.camera_temporal = self.get_parameter('camera_temporal').value
        self.temporal_buffer_size = self.get_parameter('temporal_buffer_size').value
        self.eval_radius = self.get_parameter('eval_radius').value
        
        # Create QoS profile for the subscribers with "best effort" reliability
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Create subscribers with the QoS profile
        self.groundTruth_sub = self.create_subscription(
            PointCloud2,
            groundTruth_topic,
            self.groundTruth_callback,
            qos_profile=sensor_qos,
            callback_group=sub_callback_group
        )
        
        self.dpt_sub = self.create_subscription(
            PointCloud2,
            camera_topic,
            self.dpt_callback,
            qos_profile=sensor_qos,
            callback_group=sub_callback_group
        )
        
        # Create processing timer
        self.processing_timer = self.create_timer(
            1.0 / processing_frequency,
            self.processing_timer_callback,
            callback_group=processing_timer_callback_group
        )
        
        # Create CSV writing timer
        self.csv_timer = self.create_timer(
            1.0 / csv_write_frequency,
            self.csv_timer_callback,
            callback_group=csv_timer_callback_group
        )
        
        # Initialize raw data buffers
        self.gt_object_msg = None
        self.dpt_pointcloud_msg = None
        
        # Temporal buffer for camera detections
        self.camera_temporal_buffer = deque(maxlen=self.temporal_buffer_size)
        
        # Memory buffer for groundTruth detection (1 entry)
        self.groundTruth_memory_buffer = None
        
        # Store processed detections for CSV writing
        self.latest_gt_detection = None
        self.latest_dpt_ptc = None
        self.latest_error = None
        self.has_new_data = False
        
        # Initialize CSV file
        self.initialize_csv()
        
        self.get_logger().info(
            f"PointCloud processor initialized:\n"
            f"  groundTruth topic: {groundTruth_topic}\n"
            f"  Camera topic: {camera_topic}\n"
            f"  Processing frequency: {processing_frequency} Hz\n"
            f"  CSV write frequency: {csv_write_frequency} Hz\n"
            f"  Camera aggregation mode: {self.camera_agg_mode}\n"
            f"  Camera temporal aggregation: {self.camera_temporal}\n"
            f"  Temporal buffer size: {self.temporal_buffer_size}\n"
            f"  CSV output: {self.csv_output_path}"
        )
    
    def initialize_csv(self):
        """Initialize the CSV file with headers, truncating any existing file."""
        # Always open in 'w' mode to truncate existing files
        with open(self.csv_output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp',
                'groundTruth_x', 'groundTruth_y', 'groundTruth_z', 'groundTruth_r', 'groundTruth_theta', 'groundTruth_phi',
                'camera_x', 'camera_y', 'camera_z', 'camera_r', 'camera_theta', 'camera_phi',
                'error_r',  # Radial error
                'error_euclidean'  # Euclidean error for reference
            ])
        self.get_logger().info(f"Created new CSV file: {self.csv_output_path} (truncated any existing file)")
    
    def gt_callback(self, msg):
        """Store incoming groundTruth point cloud data."""
        self.gt_object_msg = msg
        self.get_logger().debug(f"Received gt, timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
    
    def dpt_callback(self, msg):
        """Store incoming camera point cloud data."""
        self.dpt_pointcloud_msg = msg
        self.get_logger().debug(f"Received camera pointcloud, timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
    
    def cartesian_to_polar(self, x, y, z):
        """
        Convert Cartesian coordinates to polar coordinates.
        
        Args:
            x, y, z: Cartesian coordinates
            
        Returns:
            r, theta, phi: Polar coordinates
            - r: radius (distance from origin)
            - theta: azimuthal angle in x-y plane (0 to 2π)
            - phi: polar angle from z-axis (0 to π)
        """
        r = math.sqrt(x**2 + y**2 + z**2)
        
        # Handle the case where r is zero to avoid division by zero
        if r == 0:
            return 0, 0, 0
            
        theta = math.atan2(y, x)  # Azimuthal angle (0 to 2π)
        phi = math.acos(z / r)    # Polar angle from z-axis (0 to π)
        
        return r, theta, phi
    
    def polar_to_cartesian(self, r, theta, phi):
        """
        Convert polar coordinates to Cartesian coordinates.
        
        Args:
            r: radius (distance from origin)
            theta: azimuthal angle in x-y plane (0 to 2π)
            phi: polar angle from z-axis (0 to π)
            
        Returns:
            x, y, z: Cartesian coordinates
        """
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        
        return x, y, z
    
    def convert_pointcloud_to_polar(self, pointcloud_msg):
        """
        Convert PointCloud2 message to a list of points in both Cartesian and polar coordinates.
        
        Args:
            pointcloud_msg: PointCloud2 message
            
        Returns:
            List of tuples (x, y, z, r, theta, phi)
        """
        polar_points = []
        
        # Convert PointCloud2 to a list of points
        for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            r, theta, phi = self.cartesian_to_polar(x, y, z)
            polar_points.append((x, y, z, r, theta, phi))
        
        return polar_points
    
    def convert_tracked_object_to_polar(self):
        """
        Convert tracked_object_msg message to a list of points in both Cartesian and polar coordinates.
        
        Args:
            pointcloud_msg: tracked_object_msg message
            
        Returns:
            List of tuples (x, y, z, r, theta, phi)
        """
        tracked_objects_msg = self.tracked_objects_msg
        # Convert PointCloud2 to a list of points
        if len(tracked_objects_msg.objects) == 0:
            return None
        point = tracked_objects_msg.objects[0].detection.pose.position
        x = point.x
        y = point.y 
        z = point.z
        r, theta, phi = self.cartesian_to_polar(x, y, z)
        
        return (x, y, z, r, theta, phi)
    
    def euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            point1, point2: Points as (x, y, z, ...) or (x, y, z)
            
        Returns:
            Euclidean distance
        """
        # Extract just the x, y, z coordinates
        x1, y1, z1 = point1[0:3]
        x2, y2, z2 = point2[0:3]
        
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    def extract_eval_point(self):
        """
        Process camera pointcloud to get a single detection point in polar coordinates.
        
        Args:
            pointcloud_msg: PointCloud2 message from camera
            
        Returns:
            A tuple (x, y, z, r, theta, phi) representing the detection in both
            Cartesian and polar coordinates, or None if no detection is made
        """
        # Convert PointCloud2 to a list of points in polar coordinates
        dpt_polar_points = self.convert_pointcloud_to_polar(self.dpt_pointcloud_msg)
        
        if not dpt_polar_points:
            self.get_logger().warn("Camera pointcloud is empty")
            return None
        
        # If there's only one point, return it
        if len(dpt_polar_points) == 1:
            return dpt_polar_points[0]
        
        # Filter points within evaluation radius and find closest point in one pass
        groundTruth_point = self.convert_tracked_object_to_polar()
        qualified_points = []
        closest_point = None
        min_distance = float('inf')

        for point in dpt_polar_points:
            distance = self.euclidean_distance(point, groundTruth_point)
            
            # Check if point is within radius
            if distance <= self.eval_radius:
                qualified_points.append(point)
            
            # Track closest point regardless of radius
            if distance < min_distance:
                min_distance = distance
                closest_point = point

        # Use closest point if no qualified points found
        if not qualified_points:
            print(" No points within eval_radius of ground truth")
            return closest_point
        
        # Process according to aggregation mode
        if self.camera_agg_mode == "MEDIAN":
            # Sort by radius for median calculation
            qualified_points.sort(key=lambda p: p[3])  # Sort by r (index 3)
            # Get the median point based on radius
            median_idx = len(qualified_points) // 2
            detection = qualified_points[median_idx]
        elif self.camera_agg_mode == "AVERAGE":
            # Calculate the average radius
            avg_r = sum(p[3] for p in qualified_points) / len(qualified_points)
            
            # Find the point closest to the average radius
            closest_idx = min(range(len(qualified_points)), 
                              key=lambda i: abs(qualified_points[i][3] - avg_r))
            detection = qualified_points[closest_idx]
        else:
            self.get_logger().error(f"Unknown aggregation mode: {self.camera_agg_mode}")
            return None
        
        return detection
    
    def get_groundTruth(self):
        """
        Process groundTruth pointcloud based on camera detection and previous groundTruth detection.
        
        Returns:
            A tuple (x, y, z, r, theta, phi) representing the groundTruth detection,
            or None if no suitable detection is found
        """ 
        # Convert radar pointcloud to polar coordinates
        groundTruth_point = self.convert_tracked_object_to_polar()
        
        if not groundTruth_point:
            self.get_logger().warn("Tracked Obejcts is empty")
            return None
        
        # If there's only one point, return it and store in buffer
        self.groundTruth_memory_buffer = groundTruth_point
        return groundTruth_point

    def calculate_error(self, radar_detection, camera_detection):
        """
        Calculate error between radar and camera detection points.
        
        Args:
            radar_detection: (x, y, z, r, theta, phi) tuple from radar
            camera_detection: (x, y, z, r, theta, phi) tuple from camera
            
        Returns:
            Dictionary with error components
        """
        if radar_detection is None or camera_detection is None:
            return None
        
        # Extract Cartesian and polar coordinates
        radar_x, radar_y, radar_z, radar_r, radar_theta, radar_phi = radar_detection
        camera_x, camera_y, camera_z, camera_r, camera_theta, camera_phi = camera_detection
        
        # Calculate radial error (camera_r - radar_r)
        error_r = camera_r - radar_r
        
        # Calculate Euclidean distance for reference
        error_euclidean = math.sqrt(
            (radar_x - camera_x)**2 + 
            (radar_y - camera_y)**2 + 
            (radar_z - camera_z)**2
        )
        
        return {
            'error_r': error_r,
            'error_euclidean': error_euclidean
        }
    
    def processing_timer_callback(self):
        """
        Processing timer callback to process pointclouds and match detections.
        This runs at the frequency specified by the processing_frequency parameter.
        Results are stored for the CSV timer to write.
        """
        # Check if we have both pointclouds
        if self.gt_object_msg is not None and self.dpt_pointcloud_msg is not None:
            # Process camera pointcloud
            eval_point = self.extract_eval_point() # Calculate the ball here
            
            if eval_point is not None:
                # If using temporal aggregation
                if self.camera_temporal:
                    # Add to buffer - will automatically drop oldest when full
                    self.camera_temporal_buffer.append(eval_point)
                    
                    # Only process if buffer is full
                    if len(self.camera_temporal_buffer) < self.temporal_buffer_size:
                        self.get_logger().debug(
                            f"Camera temporal buffer not full yet: {len(self.camera_temporal_buffer)}/{self.temporal_buffer_size}"
                        )
                        # Clear pointcloud buffers but keep temporal buffer
                        self.gt_object_msg = None
                        self.dpt_pointcloud_msg = None
                        return
                    
                    # Process the temporal buffer based on aggregation mode
                    if self.camera_agg_mode == "MEDIAN":
                        # Sort by radius and take median
                        sorted_buffer = sorted(self.camera_temporal_buffer, key=lambda p: p[3])
                        median_idx = len(sorted_buffer) // 2
                        camera_detection = sorted_buffer[median_idx]
                    elif self.camera_agg_mode == "AVERAGE":
                        # Calculate average radius
                        avg_r = sum(p[3] for p in self.camera_temporal_buffer) / len(self.camera_temporal_buffer)
                        
                        # Use average radius with consistent direction (from last detection)
                        last_detection = self.camera_temporal_buffer[-1]
                        x, y, z = self.polar_to_cartesian(avg_r, last_detection[4], last_detection[5])
                        camera_detection = (x, y, z, avg_r, last_detection[4], last_detection[5])
                
                # Process radar pointcloud based on camera detection
                groundTruth_detection = self.get_groundTruth()
                
                # Skip if groundTruth_detection is None (memory buffer just initialized)
                if groundTruth_detection is None:
                    self.get_logger().info("Initializing radar memory buffer, skipping evaluation")
                    
                    # Clear pointcloud buffers
                    self.gt_object_msg = None
                    self.dpt_pointcloud_msg = None
                    return
                
                # Calculate error between detections
                error = self.calculate_error(groundTruth_detection, camera_detection)
                
                if error:
                    # Store the latest detections and error for CSV writing
                    self.latest_groundTruth_detection = groundTruth_detection
                    self.latest_camera_detection = camera_detection
                    self.latest_error = error
                    self.has_new_data = True
                    
                    self.get_logger().info(
                        f"Processed pointclouds. Radial error: {error['error_r']:.3f}m, "
                        f"Euclidean error: {error['error_euclidean']:.3f}m"
                    )
            else:
                self.get_logger().warn("No camera detection extracted")
        else:
            missing = []
            if self.gt_object_msg is None:
                missing.append("ground truth")
                # Drop radar memory buffer if radar pointcloud is missing
                if self.groundTruth_memory_buffer is not None:
                    self.get_logger().debug("Dropping groundTruth detection from memory buffer due to missing ground truth tracked object")
                    self.groundTruth_memory_buffer = None
            if self.dpt_pointcloud_msg is None:
                missing.append("camera")
            
            self.get_logger().debug(
                f"Skipping processing. Missing pointclouds from: {', '.join(missing)}"
            )
            
            # If using temporal aggregation and buffer has items, drop the oldest item
            if self.camera_temporal and len(self.camera_temporal_buffer) > 0:
                # Remove the oldest item from the buffer
                self.camera_temporal_buffer.popleft()
                self.get_logger().debug("Dropped oldest detection from temporal buffer due to missing pointclouds")
        
        # Clear pointcloud buffers after processing
        self.gt_object_msg = None
        self.dpt_pointcloud_msg = None
    
    def csv_timer_callback(self):
        """
        CSV timer callback to write the latest detections to the CSV file.
        This runs at the frequency specified by the csv_write_frequency parameter.
        """
        # Check if we have new data to write
        if self.has_new_data:
            self.write_to_csv(
                self.latest_radar_detection,
                self.latest_camera_detection,
                self.latest_error
            )
            self.get_logger().info("Wrote latest detections to CSV")
            self.has_new_data = False
        else:
            self.get_logger().debug("No new data to write to CSV")
    
    def write_to_csv(self, radar_detection, camera_detection, error):
        """
        Write the detections and error to the CSV file.
        
        Args:
            radar_detection: (x, y, z, r, theta, phi) tuple from radar
            camera_detection: (x, y, z, r, theta, phi) tuple from camera
            error: Dictionary with error components
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        
        with open(self.csv_output_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                radar_detection[0], radar_detection[1], radar_detection[2],
                radar_detection[3], radar_detection[4], radar_detection[5],
                camera_detection[0], camera_detection[1], camera_detection[2],
                camera_detection[3], camera_detection[4], camera_detection[5],
                error['error_r'],
                error['error_euclidean']
            ])

def main(args=None):
    rclpy.init(args=args)
    
    pointcloud_processor = PointCloudProcessor()
    
    try:
        rclpy.spin(pointcloud_processor)
    except KeyboardInterrupt:
        pass
    finally:
        pointcloud_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()