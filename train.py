import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import logging
import os

# 导入自定义模块
from dataset import DataLoader as CustomDataLoader, dual_transform, EyeDataset
from model import DualViT
from model_training import train_epoch, validate_epoch  # 更新了训练和验证的逻辑

from datetime import datetime  # 导入datetime模块

# 获取当前时间并格式化
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式化为年月日_时分秒

# 1. 创建保存模型和日志的目录
os.makedirs("model", exist_ok=True)  # 如果没有model目录，则创建
os.makedirs("log", exist_ok=True)  # 如果没有log目录，则创建

# 使用当前时间来构建模型保存路径
checkpoint_path = f"model/checkpoint_{current_time}.pth"
best_model_path = f"model/Vitmodel_{current_time}.pth"

# 2. 设置日志
logging.basicConfig(filename='log/training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 3. 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据是否有 GPU 选择设备
logging.info(f"Using device: {device}")

# 4. 加载数据
file_path = "../data_3k/Training_Dataset.xlsx"  # 数据文件路径
img_dir = "../data_3k/Training_Dataset"  # 图像文件夹路径

# 使用自定义 DataLoader 类加载数据
data_loader = CustomDataLoader(file_path, img_dir)
train_left, val_left, train_right, val_right, train_labels, val_labels = data_loader.load_data(train=True)  # 加载训练集和验证集数据

train_dataset = EyeDataset(train_left, train_right, train_labels, transform=dual_transform)  # 创建训练集数据集
val_dataset = EyeDataset(val_left, val_right, val_labels, transform=dual_transform)  # 创建验证集数据集

# 创建 DataLoader 实例，用于批量加载数据
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  # 训练集 DataLoader
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)  # 验证集 DataLoader

# 输出训练集和验证集样本数量
logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# 5. 加载模型
model = DualViT(num_classes=8).to(device)  # 8 类分类问题，加载 DualViT 模型

# 6. 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失，用于多标签分类
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)  # 使用 AdamW 优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # 学习率调度器，当验证损失不再下降时降低学习率

# 7. 加载检查点（如果有的话）
start_epoch = 1
best_val_loss = float('inf')

# 恢复模型和优化器的状态
if torch.cuda.is_available() and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从上一个epoch开始
    best_val_loss = checkpoint['best_val_loss']  # 恢复最佳验证损失
    logging.info(f"Resumed from checkpoint at epoch {start_epoch - 1}")

# 8. 训练参数
num_epochs = 20  # 训练轮数

# 9. 训练循环
for epoch in range(start_epoch, num_epochs + 1):

    # 训练一个 epoch
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)  # 训练一个 epoch
    logging.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")

    # 验证一个 epoch
    val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, num_epochs)  # 验证一个 epoch
    logging.info(f"Epoch {epoch}/{num_epochs} - Validation Loss: {val_loss:.4f}")

    # 打印当前验证集的损失和评估指标
    logging.info(f"[Val] Loss={val_loss:.4f}, Acc_all={val_metrics['acc_all']:.4f}, "
                 f"Acc_single={val_metrics['acc_single']:.4f}, Precision={val_metrics['precision']:.4f}, "
                 f"Recall={val_metrics['recall']:.4f}, f1={val_metrics['f1_score']:.4f}")

    # 如果验证损失有所改进，保存当前模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss  # 更新最佳验证损失
        torch.save(model.state_dict(), best_model_path)  # 保存模型参数
        logging.info(f"Best model saved at {best_model_path}")

    # 保存检查点（模型、优化器、调度器和epoch）
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")

    # 根据验证损失调整学习率
    scheduler.step(val_loss)
