from ultralytics import YOLO


model = YOLO("best.pt")

model.export(format="onnx", imgsz=(154, 1008)) # creates best.onnx
