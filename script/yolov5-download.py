from ultralytics import YOLO

model = YOLO("yolov5s.pt")
success = model.export(format="onnx", simplify=True)
assert success
print("转换成功")
