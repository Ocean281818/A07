import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from model import DualViT  # 导入 ViT 模型
from dataset import DataLoader as CustomDataLoader  # 使用自定义 DataLoader 类
from dataset import dual_transform, EyeDataset  # 统一使用训练时的数据增强
from evaluate import compute_metrics  # 导入计算评估指标的函数
import logging

# 1. 创建保存日志的目录
os.makedirs("log", exist_ok=True)  # 如果没有log目录，则创建

# 2. 设置日志
logging.basicConfig(filename='log/testing.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# **测试函数**
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for left_imgs, right_imgs, labels in tqdm(test_loader, desc="Testing"):
            left_imgs, right_imgs, labels = left_imgs.to(device), right_imgs.to(device), labels.to(device)
            outputs = model(left_imgs, right_imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # outputs = torch.sigmoid(outputs)

            # print("Probabilities:", probs)
            # 收集所有预测值和真实标签
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # 转换为 NumPy 数组
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # 计算评估指标
    metrics = compute_metrics(torch.tensor(all_labels), torch.tensor(all_preds),0.3)

    avg_loss = total_loss / len(test_loader)

    # 打印测试结果
    logging.info(f"\nTest Results:\n"
                 f"  Loss: {avg_loss:.4f}\n"
                 f"  Accuracy (Exact Match): {metrics['acc_all']:.4f}\n"
                 f"  Accuracy (Per-label Avg): {metrics['acc_single']:.4f}\n"
                 f"  Precision: {metrics['precision']:.4f}\n"
                 f"  Recall: {metrics['recall']:.4f}\n"
                 f"  F1-Score: {metrics['f1_score']:.4f}\n")

    return all_preds, all_labels


# 运行测试
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = 'model/Vitmodel_20250131_231535.pth'  # 使用模型保存路径
    if not os.path.exists(model_path):
        raise FileNotFoundError("未找到训练好的模型 (.pth)！请检查 `./model` 目录。")

    logging.info(f"加载模型: {model_path}")

    # 加载模型
    model = DualViT(num_classes=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))


    # 加载测试数据
    test_file = "../data_3k/Training_Dataset.xlsx"
    test_img_dir = "../data_3k/Training_Dataset"
    data_loader = CustomDataLoader(test_file, test_img_dir)
    left_paths, right_paths, labels = data_loader.load_data(train=False)

    test_dataset = EyeDataset(left_paths, right_paths, labels, transform=dual_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    logging.info(f"测试样本数: {len(test_dataset)}")

    # 测试模型
    predictions, ground_truth = test_model(model, test_loader, device)
