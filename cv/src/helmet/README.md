# 安全帽识别

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
vim voc2yolo.py

---
import os
import pickle
from os.path import join
from os import listdir, getcwd
import xml.etree.ElementTree as ET

classes = ["helmet", "person"]

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id):
    if not os.path.exists('Annotations/%s.xml' % (image_id)):
        return
    in_file = open('Annotations/%s.xml' % (image_id))
    out_file = open('labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (
            float(xmlbox.find('xmin').text),
            float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymin').text),
            float(xmlbox.find('ymax').text)
        )
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a).ljust(8, '0')[:8] for a in bb]) + '\n')
 
for image in os.listdir('JPEGImages'):
    image_id = image.split('.')[0]
    convert_annotation(image_id)
---

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
    epochs     = 100,
    single_cls = True,  
    batch      = 16,
    workers    = 10,
#   device     = 'cuda',
)
---
```
