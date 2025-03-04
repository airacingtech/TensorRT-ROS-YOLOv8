from ultralytics import YOLO


model = YOLO("best.pt")

model.export(format="onnx", imgsz=(160, 1024)) # creates best.onnx