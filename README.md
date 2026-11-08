深度学习目标检测中的基于YOLOv7框架下训练建筑工地扬尘浮土检测数据集 从数据预处理到模型可视化
@[toc]
建筑工地扬尘浮土检测数据集，3833张，yolo和voc标
1类，标注数量：
dust: 3832
image num: 3833

基于YOLOv7框架下训练您的建筑工地扬尘浮土检测数据集的步骤。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1bf0493d7b8d41d79f3e994107980b35.png)

### 数据集预处理
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7bf110a904ea405b8fa6e42c799d68ee.png)

#### 1. VOC格式转换为YOLO格式
将VOC格式的标注文件转换为YOLO所需的txt格式。每个图像对应一个txt文件，其中每一行代表一个目标，格式如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4561e7f0c2b640de85115a6a6ccac784.png)

```
class_index center_x center_y width height
```
所有坐标值均相对于图像尺寸归一化。

```python
import xml.etree.ElementTree as ET
import os

def convert_voc_to_yolo(voc_dir, yolo_dir, classes):
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)
    
    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith('.xml'):
            continue
        
        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()
        
        image_name = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        with open(os.path.join(yolo_dir, image_name.replace('.jpg', '.txt')), 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = classes.index(class_name)
                
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                w = (xmax - xmin) / float(width)
                h = (ymax - ymin) / float(height)
                
                f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

# 示例调用
convert_voc_to_yolo('path/to/voc_labels', 'path/to/yolo_labels', ['dust'])
```

### 模型训练

使用的是类似于YOLOv7的框架：

#### 安装依赖
克隆仓库并安装所需依赖。
```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements.txt
```

#### 配置数据集路径与类别信息
在`data/coco.yaml`中修改为您的数据集路径和类别信息。
```yaml
train: ./path/to/train/images
val: ./path/to/val/images

nc: 1
names: ['dust']
```

#### 训练模型
使用预训练权重开始训练。
```bash
python train.py --img 640 --batch 16 --epochs 50 --data path/to/your_data.yaml --weights yolov7.pt
```

### 模型推理与可视化

#### 推理代码示例
```python
from models.experimental import attempt_load
from utils.general import non_max_suppression
import torch
from PIL import Image
import torchvision.transforms as transforms

model = attempt_load('path/to/your_trained_model.pt', map_location='cpu')
model.eval()

image = Image.open('path/to/image.jpg')
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    pred = model(image_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

for i, det in enumerate(pred):
    # det包含预测结果
    print(det)
```

#### 可视化界面
创建一个简单的可视化界面，可以使用Streamlit框架。以下是一个基本：

```python
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
from models.experimental import attempt_load
from utils.general import non_max_suppression
import torch
from utils.plots import plot_one_box

st.title("Dust Detection Interface")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

model = attempt_load('path/to/your_trained_model.pt', map_location='cpu')
model.eval()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(image_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    
    draw = ImageDraw.Draw(image)
    for *xyxy, conf, cls in reversed(pred[0]):
        label = f'{model.names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=3)
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
```

同学，需要根据实际发布的文档和代码库进行相应的调整。
