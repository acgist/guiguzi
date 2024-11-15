# 机器视觉

|功能|路径|功能描述|
|:--|:--|:--|
|人脸检测|`face-detection`|检测人脸|
|活体检测|`face-anti-spoofing`|检测嘴巴、眼睛状态|
|图片切割|`segmentation`|图片语义切割|
|人脸识别|`face-recognition`|识别人脸相似度|
|安全帽识别|`helmet`|判断是否人物是否正确穿戴安全帽|

## ONNX

#### 下载

* https://onnxruntime.ai/getting-started
* https://github.com/microsoft/onnxruntime/releases

#### 模型

* https://blog.csdn.net/matt45m/article/details/139029398

```
conda create --name yolo python==3.10
conda activate yolo
pip install ultralytics

from ultralytics import YOLO
# yolov5s.pt
# yolov8n_face.pt
model = YOLO("yolov5s.pt")
success = model.export(format="onnx", simplify=True)
assert success
print("转换成功")
```

## YOLO

* https://docs.ultralytics.com/yolov5/tutorials

#### 训练

* https://blog.csdn.net/guanjing_dream/article/details/123715084
* https://blog.csdn.net/weixin_51639169/article/details/133859732

#### 部署

* https://github.com/yasenh/libtorch-yolov5
* https://www.cnblogs.com/kods/p/15119831.html
* https://cloud.tencent.com/developer/article/2366473
* https://blog.csdn.net/lucifer479/article/details/120743124

#### 模型

* yolov5s.pt
* yolov8n_face.pt
* yolo11n.pt
* yolo11s.pt
* yolo11m.pt
* yolo11x.pt

## LibTorch

```
# GPU->CPU
model.to("cpu")

# CPU->GPU
model.to("cuda")

# 整个模型
torch.save(model, PATH)
model = torch.load(PATH)

# 模型参数：需要结构
torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))

# onnx
model.export(format="onnx", simplify=True)

# libtorch
model.export(format="torchscript")

# onnx
torch.onnx.export(model.eval(), input, PATH)

# trace
model = torch.jit.trace(model.eval(), input)
model.save(PATH)

# script
model = torch.jit.script(model.eval())
model.save(PATH)
```
