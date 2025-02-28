# import os
# import glob
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np

# # Path to your image folder - replace with your path
# IMAGE_FOLDER = "test_image"  # REPLACE THIS WITH YOUR PATH
# SAVE_FOLDER = "results"  # Where to save the visualization results
# MODEL_PATH = "best.pt"  # Path to your model

# # Target aspect ratio and dimensions
# TARGET_HEIGHT = 154
# TARGET_WIDTH = 1008
# TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT

# # Create save folder if it doesn't exist
# os.makedirs(SAVE_FOLDER, exist_ok=True)

# # Load the YOLO model
# model = YOLO(MODEL_PATH)

# # Recursively get all image files from the folder and subfolders
# image_files = []
# for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
#     image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, '**', ext), recursive=True))

# print(f"Found {len(image_files)} images in folder and subfolders")

# def crop_to_aspect_ratio(image, target_aspect_ratio):
#     """
#     Crop the image to match the target aspect ratio while keeping the center of the image.
#     """
#     height, width = image.shape[:2]
#     current_aspect_ratio = width / height
    
#     if current_aspect_ratio > target_aspect_ratio:
#         # Image is wider than target aspect ratio, crop width
#         new_width = int(height * target_aspect_ratio)
#         start_x = (width - new_width) // 2
#         cropped = image[:, start_x:start_x + new_width]
#     else:
#         # Image is taller than target aspect ratio, crop height
#         new_height = int(width / target_aspect_ratio)
#         start_y = (height - new_height) // 2
#         cropped = image[start_y:start_y + new_height, :]
    
#     return cropped

# def resize_image(image, target_width, target_height):
#     """
#     Resize the image to target dimensions.
#     """
#     return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

# # Process each image
# for img_path in image_files:
#     # Get base filename without extension
#     base_name = os.path.basename(img_path).split('.')[0]
    
#     # Read original image
#     original = cv2.imread(img_path)
    
#     if original is None:
#         print(f"Error reading image: {img_path}")
#         continue
        
#     original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
#     # Crop the image to target aspect ratio
#     cropped_image = crop_to_aspect_ratio(original, TARGET_ASPECT_RATIO)
    
#     # Resize to target dimensions
#     resized_image = resize_image(cropped_image, TARGET_WIDTH, TARGET_HEIGHT)
    
#     # Save the cropped and resized image with a new filename
#     preprocessed_path = os.path.join(SAVE_FOLDER, f"{base_name}_preprocessed.jpg")
#     cv2.imwrite(preprocessed_path, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
    
#     # Run prediction on the preprocessed image
#     results = model(preprocessed_path)
#     result = results[0]  # Get the first (only) result
    
#     # Optionally save the segmentation result
#     result_path = os.path.join(SAVE_FOLDER, f"{base_name}_segmentation.jpg")
#     result.save(filename=result_path)
    
#     print(f"Processed and saved: {preprocessed_path} and {result_path}")

# print("All images processed.")

import os
import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import onnxruntime as ort

# Path to your image folder - replace with your path
IMAGE_FOLDER = "test_image"  # REPLACE THIS WITH YOUR PATH
SAVE_FOLDER = "results"  # Where to save the visualization results
MODEL_PATH = "best.pt"  # Path to your model
ONNX_MODEL_PATH = "best.onnx"  # Path to save the ONNX model

# Target aspect ratio and dimensions
TARGET_HEIGHT = 160
TARGET_WIDTH = 1024
TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT

# Create save folder if it doesn't exist
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Load the YOLO model
model = YOLO(MODEL_PATH)

# Export the model to ONNX format
print(f"Exporting model to ONNX format: {ONNX_MODEL_PATH}")
model.export(format="onnx", imgsz=(TARGET_HEIGHT, TARGET_WIDTH))

# Load the ONNX model for inference
print("Loading ONNX model for inference")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# Recursively get all image files from the folder and subfolders
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
    image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, '**', ext), recursive=True))

print(f"Found {len(image_files)} images in folder and subfolders")

def crop_to_aspect_ratio(image, target_aspect_ratio):
    """
    Crop the image to match the target aspect ratio while keeping the center of the image.
    """
    height, width = image.shape[:2]
    current_aspect_ratio = width / height
    
    if current_aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio, crop width
        new_width = int(height * target_aspect_ratio)
        start_x = (width - new_width) // 2
        cropped = image[:, start_x:start_x + new_width]
    else:
        # Image is taller than target aspect ratio, crop height
        new_height = int(width / target_aspect_ratio)
        start_y = (height - new_height) // 2
        cropped = image[start_y:start_y + new_height, :]
    
    return cropped

def resize_image(image, target_width, target_height):
    """
    Resize the image to target dimensions.
    """
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def preprocess_for_onnx(image):
    """
    Preprocess image for ONNX model inference.
    
    Returns:
        np.ndarray: Preprocessed image in the format expected by the ONNX model
    """
    # Resize if needed
    if image.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
        image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # Convert to RGB if not already
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Transpose from HWC to NCHW format (batch, channels, height, width)
    image = image.transpose(2, 0, 1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def postprocess(input_image, output):
    """
    Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

    Args:
        input_image (numpy.ndarray): The input image.
        output (numpy.ndarray): The output of the model.

    Returns:
        numpy.ndarray: The input image with detections drawn on it.
    """
    # Transpose and squeeze the output to match the expected shape
    outputs = np.transpose(np.squeeze(output[0]))

    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor = 1
    y_factor = 1

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]

        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= 0.3:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.3)

    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        # Draw the detection on the input image
        self.draw_detections(input_image, box, score, class_id)

    # Return the modified input image
    return input_image

# Process each image
for img_path in image_files:
    # Get base filename without extension
    base_name = os.path.basename(img_path).split('.')[0]
    
    # Read original image
    original = cv2.imread(img_path)
    
    if original is None:
        print(f"Error reading image: {img_path}")
        continue
        
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Crop the image to target aspect ratio
    cropped_image = crop_to_aspect_ratio(original, TARGET_ASPECT_RATIO)
    
    # Resize to target dimensions
    resized_image = resize_image(cropped_image, TARGET_WIDTH, TARGET_HEIGHT)
    
    # Save the cropped and resized image with a new filename
    preprocessed_path = os.path.join(SAVE_FOLDER, f"{base_name}_preprocessed.jpg")
    cv2.imwrite(preprocessed_path, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
    
    # Preprocess image for ONNX inference
    onnx_input = preprocess_for_onnx(resized_image)
    
    # Get input and output names from the ONNX model
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]
    
    # Run ONNX inference
    onnx_outputs = ort_session.run(output_names, {input_name: onnx_input})
    
    resized_image = postprocess(resized_image, onnx_outputs)
    
    # Process ONNX outputs - format depends on your model's output structure
    # For segmentation models, typically the first output contains the predictions
    # onnx_predictions = onnx_outputs[0]
    
    # Convert ONNX predictions to visualization
    # This part depends on your model type (detection, segmentation, etc.)
    # For demonstration, we'll save the preprocessed image and visualize the first output channel
    
    # Save the preprocessed image
    preprocessed_path = os.path.join(SAVE_FOLDER, f"{base_name}_preprocessed.jpg")
    cv2.imwrite(preprocessed_path, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
    
    # # Create a simple visualization of the first prediction channel (adapt based on your model output)
    # if len(onnx_predictions.shape) >= 3:  # Segmentation mask-like output
    #     # Get the first output channel from the first item in the batch
    #     pred_mask = onnx_predictions[0, 0]  # Adjust based on your model's output format
        
    #     # Normalize mask for visualization
    #     if pred_mask.max() > 0:
    #         pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min()) * 255
    #     pred_mask = pred_mask.astype(np.uint8)
        
    #     # Apply color map for better visualization
    #     pred_mask_colored = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
        
    #     # Save the prediction visualization
    #     result_path = os.path.join(SAVE_FOLDER, f"{base_name}_segmentation.jpg")
    #     cv2.imwrite(result_path, pred_mask_colored)
    
    print(f"Processed and saved: {preprocessed_path} and {preprocessed_path}")

print("All images processed.")