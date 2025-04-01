from ultralytics import YOLO


model = YOLO("best.pt")

model.export(format="onnx", imgsz=(1056, 1056)) # creates best.onnx