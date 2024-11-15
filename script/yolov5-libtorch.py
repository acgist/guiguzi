import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s")
torch.jit.script(model).save("./yolov5s.jit.pt")
assert success
print("转换成功")
