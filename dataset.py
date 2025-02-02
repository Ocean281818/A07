"""
Author: Haii
Usage: Data Augmentation for binocular medical images.
"""

import torchvision.transforms as transforms  # 导入 torchvision 的图像处理模块
import random  # 导入 random 模块，用于生成随机数
from PIL import Image  # 导入 Pillow 库，用于图像处理
import torch  # 导入 PyTorch 库，用于深度学习
from torch.utils.data import Dataset  # 导入 PyTorch 中的 Dataset 类，用于自定义数据集
import pandas as pd  # 导入 pandas 库，用于数据处理
from sklearn.model_selection import train_test_split  # 导入用于数据集划分的工具
from pathlib import Path  # 导入 Path，用于处理文件路径
import os
import torchvision.transforms as transforms
from PIL import Image

class DualTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, left_img, right_img):

        return self.transform(left_img), self.transform(right_img)

# 定义基础的图像预处理和增强流水线
base_transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为 Tensor 格式
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224
    transforms.RandomRotation(degrees=5),  # 随机旋转图像
    transforms.ColorJitter(brightness=0.4),  # 随机调整亮度
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化图像
])

# 创建 DualTransform 对象，传入数据增强方法
dual_transform = DualTransform(base_transform)

class EyeDataset(Dataset):  # 自定义数据集类，用于加载眼部图像数据
    def __init__(self, left_eye_paths, right_eye_paths, labels, transform=None):
        self.left_eye_paths = left_eye_paths  # 存储左眼图像路径
        self.right_eye_paths = right_eye_paths  # 存储右眼图像路径
        self.labels = labels  # 存储图像对应的标签
        self.transform = transform  # 存储传入的图像变换方法

    def __len__(self):
        return len(self.labels)  # 返回数据集的长度，即图像的数量

    def __getitem__(self, index):
        left_img = Image.open(self.left_eye_paths[index]).convert('RGB')  # 打开并转换左眼图像为 RGB 格式
        right_img = Image.open(self.right_eye_paths[index]).convert('RGB')  # 打开并转换右眼图像为 RGB 格式
        label = torch.tensor(self.labels[index], dtype=torch.float32)  # 将标签转换为 Tensor 格式
        if self.transform:  # 如果传入了数据增强方法
            left_img, right_img = self.transform(left_img, right_img)  # 对左右眼图像同时进行变换
        return left_img, right_img, label  # 返回处理后的左右眼图像和标签

class DataLoader:
    def __init__(self, file_path, img_dir):
        self.file_path = Path(file_path)
        self.img_dir = Path(img_dir)
        self.data = pd.read_excel(self.file_path)
        self.label_columns =  ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

        self.data.dropna(subset=['Left-Fundus', 'Right-Fundus'], inplace=True)
        self.left_paths = [self.img_dir / str(p) for p in self.data['Left-Fundus']]
        self.right_paths = [self.img_dir / str(p) for p in self.data['Right-Fundus']]
        self.labels = self.data[self.label_columns].values

    def load_data(self, train=True, test_size=0.2, random_seed=42):
        if train:
            return train_test_split(self.left_paths,self.right_paths,self.labels,test_size=test_size,random_state=random_seed)
        else:
            return self.left_paths, self.right_paths, self.labels

