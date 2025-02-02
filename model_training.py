"""
Author: Haii
用于双目医学图像分类的 ViT 训练、验证和测试函数集成。
"""
import torch  # 导入 PyTorch 库，用于深度学习
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
from evaluate import compute_metrics  # 导入自定义的指标计算函数

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()  # 设置模型为训练模式，启用 dropout 和 batch normalization
    total_loss = 0  # 初始化记录总损失的变量
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)  # 初始化进度条

    for batch_idx, (left, right, labels) in enumerate(progress_bar):  # 遍历训练集中的每个批次
        left, right, labels = left.to(device), right.to(device), labels.to(device)  # 将数据移动到指定设备（GPU 或 CPU）

        optimizer.zero_grad()  # 清空之前的梯度信息
        outputs = model(left, right)  # 将左右眼图像输入模型，获取模型输出
        loss = criterion(outputs, labels)  # 计算输出与真实标签之间的损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型的参数

        total_loss += loss.item()  # 累加当前批次的损失
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前的学习率
        progress_bar.set_postfix({  # 更新进度条的显示内容
            'Loss': loss.item(),
            'Avg Loss': total_loss / (batch_idx + 1),  # 显示当前的平均损失
            'learning_rate': f"{current_lr:.6f}"  # 显示当前的学习率
        })

    return total_loss / len(train_loader)  # 返回训练集的平均损失

def validate_epoch(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval()  # 设置模型为评估模式，禁用 dropout 和 batch normalization
    total_loss = 0  # 初始化损失变量
    all_preds, all_labels = [], []  # 初始化列表，用于存储所有预测值和标签
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} [Valid]", leave=False)  # 初始化进度条

    with torch.no_grad():  # 在验证过程中禁用梯度计算，节省内存并加速推理
        for left, right, labels in progress_bar:  # 遍历验证数据
            left, right, labels = left.to(device), right.to(device), labels.to(device)  # 将数据移动到指定设备

            outputs = model(left, right)  # 将左右眼图像输入模型，获取输出
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()  # 累加损失

            all_preds.append(outputs)  # 保存预测值
            all_labels.append(labels)  # 保存真实标签

    y_true = torch.cat(all_labels, dim=0)  # 将所有标签合并为一个 tensor
    y_pred = torch.cat(all_preds, dim=0)  # 将所有预测值合并为一个 tensor
    metrics = compute_metrics(y_true, y_pred)  # 计算评估指标（如准确率、精确率等）

    return total_loss / len(val_loader), metrics  # 返回验证集的平均损失和评估指标

def test_epoch(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    all_preds = []  # 初始化列表，用于保存所有的预测结果
    all_labels = []  # 初始化列表，用于保存所有的真实标签
    total_loss = 0  # 初始化损失变量
    criterion = torch.nn.BCEWithLogitsLoss()  # 使用二分类交叉熵损失函数（适用于二分类任务）

    with torch.no_grad():  # 禁用梯度计算，提高推理速度
        for left_imgs, right_imgs, labels in tqdm(test_loader, desc="Testing"):  # 遍历测试数据
            left_imgs, right_imgs, labels = left_imgs.to(device), right_imgs.to(device), labels.to(device)  # 将数据移动到指定设备

            outputs = model(left_imgs, right_imgs)  # 将左右眼图像输入模型，获取输出
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()  # 累加损失

            all_preds.append(outputs)  # 保存预测值
            all_labels.append(labels)  # 保存真实标签

    all_preds = torch.cat(all_preds, dim=0)  # 合并所有预测值
    all_labels = torch.cat(all_labels, dim=0)  # 合并所有真实标签
    metrics = compute_metrics(all_labels, all_preds)  # 计算评估指标

    avg_loss = total_loss / len(test_loader)  # 计算测试集的平均损失

    print(f"\nTest Results:\n"
          f"  Accuracy(每行完全匹配): {metrics['acc_all']:.4f}\n"  # 打印每行完全匹配的准确率
          f"  Accuracy: {metrics['acc_single']:.4f}\n"  # 打印单个标签的准确率
          f"  Precision: {metrics['precision']:.4f}\n"  # 打印精准率
          f"  Recall: {metrics['recall']:.4f}\n"  # 打印召回率
          f"  F1-Score: {metrics['f1_score']:.4f}\n"  # 打印 F1 分数
          )

    return all_preds, all_labels  # 返回所有的预测值和真实标签
