import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    # 确保输入为NumPy数组（如果是Torch张量，先转CPU再转NumPy）
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # 二值化处理
    y_pred_bin = (y_pred > threshold).astype(np.float32)

    # 计算指标
    acc_all = (y_true == y_pred_bin).all(axis=1).mean()
    acc_single = (y_true.flatten() == y_pred_bin.flatten()).mean()

    # 多标签分类指标
    precision = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred_bin, average='samples', zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)

    return {
        "acc_all": acc_all,
        "acc_single": acc_single,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

