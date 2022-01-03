# 与数据处理相关的函数
import os
import torch
import numpy as np
from PIL import Image
from xml.dom.minidom import parse
import cv2
from pathlib import Path
import glob,time
from utils.utils import letterbox
from threading import Thread
PATH = "data/"


# 生成用来训练目标检测模型的数据集---resnet内核版本，本机训练
class DataLoader(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 加载标注好的所有图片，使用sorted来确保这些图片都是一致大小的
        self.images = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))

    def __getitem__(self, index):
        # 加载图片和bbox
        img_path = os.path.join(self.root, "JPEGImages", self.images[index])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[index])
        img = Image.open(img_path).convert("RGB")

        # 读取文件，VOC格式的数据集的标注是xml格式的文件
        dom = parse(bbox_xml_path)

        # 获取文档元素对象
        data = dom.documentElement

        # 获取 objects
        objects = data.getElementsByTagName('object')

        # 获取边界框的坐标
        boxes = []
        labels = []
        for object_ in objects:
            # 获取标签中内容
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # 就是label
            labels.append(np.int(name[-1]))  # 背景的label是0，mark_type的label是1
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)


# 生成用来训练目标检测模型的数据集---YOLO模型
class LoadImages:
    # 为了加载推理数据集
    def __init__(self, path, img_size=640):
        self.path = str(Path(path))
        files = []
        if os.path.isdir(self.path):
            files = sorted(glob.glob(os.path.join(self.path, '*.*')))
        elif os.path.isfile(self.path):
            files = [self.path]
        self.img_size = img_size
        self.files = files
        self.img_num = len(files)       # 图片数量，完整数据集为75
        self.mode = 'images'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.img_num:
            raise StopIteration
        path = self.files[self.count]

        # 读取图片
        self.count += 1
        img0 = cv2.imread(path)
        assert img0 is not None, 'Image Not Found ' + path
        print('image %g/%g %s: \n' % (self.count, self.img_num, path), end='')

        # 填充调整图片大小
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return path, img, img0

    def __len__(self):
        return self.img_num  # number of files






