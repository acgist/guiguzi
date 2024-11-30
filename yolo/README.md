# 转为YOLO数据

## 安全帽数据集

* https://aistudio.baidu.com/datasetdetail/63355
* https://pan.baidu.com/s/1UbFkGm4EppdAU660Vu7SdQ

## 安全带数据集

* https://aistudio.baidu.com/datasetdetail/93034/0

## 标记数据

## 模型训练

```
# 下载代码
mkdir -p /data/yolo
cd /data/yolo
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics

# 标记数据
数据集：https://pan.baidu.com/s/1UbFkGm4EppdAU660Vu7SdQ

# 转换标记
Csv2YoloTest.java
Voc2YoloTest.java

# 配置文件
vim data.yaml

---
train: helmet/train
val  : helmet/val

names:
  0: helmet
  1: person
---

vim ultralytics/cfg/models/11/yolo11.yaml

---
nc: 2
---

# 开始训练
vim train.py

---
from ultralytics import YOLO
 
model = YOLO('yolo11n.pt')
model.train(
    data       = '/data/yolo/ultralytics/data.yaml',
    imgsz      = 640,
    epochs     = 200,
    single_cls = False,  
    batch      = 16,
    workers    = 16,
#   device     = 'cuda',
)
model.export(format="onnx")
---

python3 train.py

# 模型转换

---
from ultralytics import YOLO
 
model = YOLO('./best.pt')
model.export(format="onnx")
---

# 模型预测

---
from ultralytics import YOLO

model = YOLO('./best.pt')
model("./1.jpg")[0].show()
model("./1.jpg")[0].save("./2.jpg")
---

# 图片处理

---
import os
import cv2
 
 
dataDir = "source/"
saveDir = "target/"

if not os.path.exists(saveDir):
    os.makedirs(saveDir)
 
for image in os.listdir(dataDir):
    print(image)
    input_path  = dataDir + image
    input_image = cv2.imread(input_path)
    output_path = saveDir + image
    cv2.imwrite(output_path, input_image)
---
```
