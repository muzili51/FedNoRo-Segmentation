import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import copy
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
import random

def add_morphological_noise_to_mask(mask, min_noise_pixels=5, max_noise_pixels=10):
    """
    给掩码添加5-10像素的腐蚀或膨胀噪声

    Args:
        mask: 输入掩码，numpy数组或torch.Tensor
        min_noise_pixels: 最小腐蚀/膨胀像素数量
        max_noise_pixels: 最大腐蚀/膨胀像素数量

    Returns:
        添加噪声后的掩码
    """
    import numpy as np

    # 确保输入是numpy数组
    if isinstance(mask, torch.Tensor):
        is_tensor = True
        device = mask.device
        mask_np = mask.cpu().numpy().copy()
    else:
        is_tensor = False
        mask_np = mask.copy()

    # 随机选择腐蚀或膨胀操作
    operation = random.choice(['erosion', 'dilation'])

    # 随机选择腐蚀/膨胀的像素数量
    kernel_size = random.randint(min_noise_pixels, max_noise_pixels)

    # 为每个类别分别处理
    unique_classes = np.unique(mask_np)
    noisy_mask = mask_np.copy()

    # 对每个类别创建二值掩码并应用形态学操作
    for class_val in unique_classes:
        if class_val == 0:  # 跳过背景类
            continue

        # 创建当前类别的二值掩码
        binary_mask = (mask_np == class_val).astype(np.uint8)

        # 创建结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))

        # 应用腐蚀或膨胀
        if operation == 'erosion':
            processed_mask = cv2.erode(binary_mask, kernel, iterations=1)
        else:  # dilation
            processed_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # 更新掩码：腐蚀会减少区域，膨胀会扩大区域
        if operation == 'erosion':
            # 腐蚀：从原区域中移除
            mask_to_remove = (binary_mask == 1) & (processed_mask == 0)
            noisy_mask[mask_to_remove] = 0  # 通常设为背景或其他类别
        else:  # dilation
            # 膨胀：扩展到相邻区域
            mask_to_add = (processed_mask == 1) & (binary_mask == 0)
            noisy_mask[mask_to_add] = class_val

    # 转换回原始格式
    if is_tensor:
        result = torch.from_numpy(noisy_mask).to(device)
    else:
        result = noisy_mask

    return result


class NoisyImageMaskDataset(Dataset):
    def __init__(self, image_folder, mask_folder, noise_ratio=0.0,
                 min_noise_pixels=5, max_noise_pixels=10,
                 transform=None, target_transform=None):
        """
        Args:
            noise_ratio: 添加噪声的掩码比例 (0.0-1.0)
            min_noise_pixels: 最小腐蚀/膨胀像素数
            max_noise_pixels: 最大腐蚀/膨胀像素数
        """
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.noise_ratio = noise_ratio
        self.min_noise_pixels = min_noise_pixels
        self.max_noise_pixels = max_noise_pixels
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = self._get_sorted_files(image_folder)
        self.mask_files = self._get_sorted_files(mask_folder)

        assert len(self.image_files) == len(self.mask_files), \
            f"images nums ({len(self.image_files)}) with masks nums ({len(self.mask_files)}) imcompitable"

        self._verify_file_matching()

    def _get_sorted_files(self, folder):
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        files = []

        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(file)

        return sorted(files)

    def _verify_file_matching(self):
        image_names = [os.path.splitext(f)[0] for f in self.image_files]
        mask_names = [os.path.splitext(f)[0] for f in self.mask_files]

        for i, (img_name, mask_name) in enumerate(zip(image_names, mask_names)):
            clean_mask_name = mask_name.replace('_mask', '').replace('_label', '')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask_resized = np.array(mask.resize((256, 256), Image.NEAREST))
            # 对调整大小后的掩码也进行映射
            mapped_mask_resized = np.zeros_like(mask_resized)
            mapped_mask_resized[mask_resized == 0] = 0
            mapped_mask_resized[mask_resized == 128] = 1
            mapped_mask_resized[mask_resized == 255] = 2
            mask = torch.from_numpy(mapped_mask_resized).long()

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()

        # 按指定比例添加腐蚀/膨胀噪声
        if random.random() < self.noise_ratio:
            mask = add_morphological_noise_to_mask(mask, self.min_noise_pixels, self.max_noise_pixels)

        return image, mask


class ImageMaskDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None, target_transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = self._get_sorted_files(image_folder)
        self.mask_files = self._get_sorted_files(mask_folder)
        assert len(self.image_files) == len(self.mask_files), \
            f"images nums ({len(self.image_files)}) with masks nums ({len(self.mask_files)}) imcompitable"

        self._verify_file_matching()

    def _get_sorted_files(self, folder):
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        files = []

        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(file)

        return sorted(files)

    def _verify_file_matching(self):
        image_names = [os.path.splitext(f)[0] for f in self.image_files]
        mask_names = [os.path.splitext(f)[0] for f in self.mask_files]

        for i, (img_name, mask_name) in enumerate(zip(image_names, mask_names)):
            clean_mask_name = mask_name.replace('_mask', '').replace('_label', '')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask_resized = np.array(mask.resize((256, 256), Image.NEAREST))
            # 对调整大小后的掩码也进行映射
            mapped_mask_resized = np.zeros_like(mask_resized)
            mapped_mask_resized[mask_resized == 0] = 0
            mapped_mask_resized[mask_resized == 128] = 1
            mapped_mask_resized[mask_resized == 255] = 2
            mask = torch.from_numpy(mapped_mask_resized).long()

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def calculate_accuracy(pred, target, num_classes=3):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum().item()
    total = target.numel()
    accuracy = correct / total
    return accuracy


def calculate_iou(pred, target, num_classes=3):
    pred = torch.argmax(pred, dim=1)
    iou_list = []

    for i in range(num_classes):
        pred_i = (pred == i)
        target_i = (target == i)

        intersection = (pred_i & target_i).sum().item()
        union = (pred_i | target_i).sum().item()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        iou_list.append(iou)

    mean_iou = sum(iou_list) / len(iou_list)
    return mean_iou, iou_list


def evaluate_model(model, test_loader, device):
    model.eval()
    model = model.to(device)
    total_accuracy = 0
    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            accuracy = calculate_accuracy(outputs, masks)
            total_accuracy += accuracy

            mean_iou, _ = calculate_iou(outputs, masks)
            total_iou += mean_iou

            num_batches += 1

    avg_accuracy = total_accuracy / num_batches
    avg_iou = total_iou / num_batches

    return avg_accuracy, avg_iou

# -------------------- 对称交叉熵损失 (SL) --------------------
class SCELoss(nn.Module):
    """
    Symmetric Cross Entropy Loss: lambda * CE + RCE
    """
    def __init__(self, alpha=0.1, beta=1.0, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, pred, labels):
        # pred: logits, shape (batch_size, num_classes, height, width)
        # labels: long tensor, shape (batch_size, height, width)

        # 计算标准交叉熵损失
        ce = F.cross_entropy(pred, labels, reduction='mean')

        # 计算 RCE 损失
        pred_softmax = F.softmax(pred, dim=1)  # shape: (batch_size, num_classes, height, width)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)

        # 将 labels 转换为 one-hot 编码并调整维度
        label_one_hot = F.one_hot(labels, self.num_classes).float()  # shape: (batch_size, height, width, num_classes)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_one_hot = label_one_hot.permute(0, 3, 1, 2)  # 调整为 (batch_size, num_classes, height, width)

        # 计算 RCE: -sum(q * log(p))
        rce = (-torch.sum(pred_softmax * torch.log(label_one_hot), dim=1)).mean()

        return self.alpha * ce + self.beta * rce


# -------------------- 公共数据集 (只返回图像) --------------------
class PublicDataset(Dataset):
    """只返回图像，不返回标签，用于知识蒸馏"""
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = self._get_sorted_files(image_folder)

    def _get_sorted_files(self, folder):
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        files = []
        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(file)
        return sorted(files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# -------------------- 客户端类 --------------------
class Client:
    def __init__(self, client_id, model, train_loader, val_loader, public_loader, device):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.public_loader = public_loader  # 公共数据加载器
        self.device = device
        self.optimizer = None  # 将在训练时创建

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_local(self, epochs=5, learning_rate=0.001):
        """
        在客户端本地使用SL损失训练模型
        """
        self.model.to(self.device)
        self.model.train()
        criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=self.model.n_classes)
        if self.optimizer is None:
            self.set_optimizer(learning_rate)
        else:
            # 更新学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

        running_loss = 0.0
        for epoch in range(epochs):
            for images, masks in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                masks = torch.clamp(masks, 0, self.model.n_classes - 1)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def compute_logits_on_public(self):
        """
        计算当前模型在公共数据集上的logits
        返回: 形状为 (N_public, n_classes) 的张量
        """
        self.model.eval()
        logits_list = []
        with torch.no_grad():
            for images in self.public_loader:
                images = images.to(self.device)
                logits = self.model(images)
                logits_list.append(logits.cpu())
        return torch.cat(logits_list, dim=0)

    def distill_update(self, other_logits_list, weights, public_images, lr):
        """
        使用加权KL散度更新模型
        other_logits_list: 其他客户端的logits列表（固定，不梯度）
        weights: 对应其他客户端的权重
        public_images: 公共数据图像张量 (N, C, H, W)
        lr: 学习率
        """
        self.model.train()
        if self.optimizer is None:
            self.set_optimizer(lr)
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.optimizer.zero_grad()
        # 计算当前模型在公共数据上的logits
        current_logits = self.model(public_images.to(self.device))  # (N, n_classes)
        total_loss = 0.0
        for j, logits_j in enumerate(other_logits_list):
            # 计算 KL(R_j || R_current)
            # input: log prob of current, target: prob of j
            kl = F.kl_div(
                F.log_softmax(current_logits, dim=1),
                F.softmax(logits_j.to(self.device), dim=1),
                reduction='batchmean'
            )
            total_loss += weights[j] * kl
        total_loss.backward()
        self.optimizer.step()

    def evaluate_local(self):
        """在客户端本地评估模型"""
        return evaluate_model(self.model, self.val_loader, self.device)


# -------------------- RHFL服务器类 --------------------
class RHFLServer:
    def __init__(self, clients, public_loader, device, eta=0.5):
        self.clients = clients
        self.public_loader = public_loader
        self.device = device
        self.eta = eta  # CCR公式中的η
        self.prev_losses = [None] * len(clients)  # 存储上一轮本地损失

    def select_clients(self, fraction=1.0):
        """随机选择一部分客户端参与本轮训练"""
        num_clients = len(self.clients)
        num_selected = max(1, int(fraction * num_clients))
        selected_indices = np.random.choice(num_clients, num_selected, replace=False)
        return [self.clients[i] for i in selected_indices]

    def compute_ccr_weights(self, current_losses):
        """
        根据当前损失和上一轮损失计算CCR权重 W_k
        current_losses: 列表，每个客户端的当前本地训练平均损失
        返回: 归一化后的权重列表 W_k (softmax)
        """
        F_list = []
        for i, loss in enumerate(current_losses):
            prev = self.prev_losses[i]
            if prev is None:
                # 第一轮没有上一轮损失，设下降率为0
                delta = 0.0
            else:
                delta = prev - loss  # 损失下降量，正值表示下降
            # 标签质量 Q = 1 / loss (加小常数防止除零)
            Q = 1.0 / (loss + 1e-8)
            F = Q * max(delta, 0.0)  # 下降率取非负
            F_list.append(F)

        # 计算 w_k
        K = len(self.clients)
        w = []
        for i, F in enumerate(F_list):
            wi = 1.0 / (K - 1) + self.eta * (F / (sum(F_list) + 1e-8))
            w.append(wi)
        # softmax 归一化
        w_tensor = torch.tensor(w)
        W = torch.softmax(w_tensor, dim=0).tolist()
        return W

    def collaborative_learning_step(self, selected_clients, public_images, lr, weights):
        """
        执行一轮协作学习：每个选中客户端根据其他客户端的加权logits更新模型
        public_images: 公共数据图像张量 (N, C, H, W)
        weights: 每个客户端的CCR权重列表（长度=总客户端数，但只对选中客户端有效？实际上应该对所有客户端）
        """
        # 首先计算所有选中客户端的logits（使用更新前的模型）
        logits_dict = {}
        for client in selected_clients:
            client.model.eval()
            with torch.no_grad():
                logits = client.model(public_images.to(client.device)).cpu()
            logits_dict[client.client_id] = logits

        # 为每个选中客户端进行蒸馏更新
        for client in selected_clients:
            # 收集其他客户端的logits和对应权重
            other_logits = []
            other_weights = []
            for other in selected_clients:
                if other.client_id == client.client_id:
                    continue
                other_logits.append(logits_dict[other.client_id])
                other_weights.append(weights[other.client_id])  # 注意weights索引是client_id
            if not other_logits:
                continue  # 只有一个客户端时不更新
            # 归一化权重（使其和为1）可选，但原论文未明确，可以保留原始权重
            # 但为了稳定，可以归一化
            w_sum = sum(other_weights)
            if w_sum > 0:
                other_weights = [w / w_sum for w in other_weights]
            client.distill_update(other_logits, other_weights, public_images, lr)

    def federated_train(self, rounds=50, local_epochs=5, client_fraction=1.0,
                        learning_rate=0.001):
        """
        RHFL训练主循环
        """
        print(f"开始RHFL训练，共 {rounds} 轮")

        # 预先加载所有公共数据到一个张量（假设公共数据量不大）
        public_images_list = []
        for batch in self.public_loader:
            public_images_list.append(batch)
        public_images = torch.cat(public_images_list, dim=0)  # (N, C, H, W)

        history = {'acc': [], 'iou': []}

        for round_num in range(rounds):
            print(f"\n=== RHFL 第 {round_num + 1} 轮 ===")
            selected_clients = self.select_clients(client_fraction)
            print(f"选择 {len(selected_clients)} 个客户端参与")

            # 本地训练阶段
            current_losses = []
            for client in selected_clients:
                print(f"客户端 {client.client_id} 本地训练...")
                loss = client.train_local(epochs=local_epochs, learning_rate=learning_rate)
                current_losses.append(loss)
                # 可选：评估本地模型
                acc, iou = client.evaluate_local()
                print(f"  本地评估 - 准确率: {acc:.4f}, IoU: {iou:.4f}")

            # 计算CCR权重（需要所有客户端的损失，不仅是选中的，但未选中的本轮无损失，可用上一轮损失代替？）
            # 简化：只考虑选中客户端，未选中的置信度不参与加权，因为协作只发生在选中客户端之间
            # 我们需要为每个选中客户端分配权重，权重基于其自身损失计算
            # 但公式中权重是每个客户端自身的，应该用所有客户端计算，但未选中的损失未知，可暂时忽略或沿用上次
            # 这里我们只基于选中的客户端计算权重
            # 构建当前损失列表（与clients顺序对应）
            full_current = [None] * len(self.clients)
            for idx, client in enumerate(selected_clients):
                full_current[client.client_id] = current_losses[idx]
            # 对于未选中的，用上一轮损失代替（若无则设为一个默认值）
            for i in range(len(self.clients)):
                if full_current[i] is None:
                    full_current[i] = self.prev_losses[i] if self.prev_losses[i] is not None else 1.0

            W = self.compute_ccr_weights(full_current)  # W长度=总客户端数
            # 更新prev_losses
            self.prev_losses = full_current.copy()

            # 协作学习阶段
            print("执行协作学习...")
            self.collaborative_learning_step(selected_clients, public_images, learning_rate, W)

            # 评估选中客户端的模型（可选）
            avg_acc = 0
            avg_iou = 0
            for client in selected_clients:
                acc, iou = client.evaluate_local()
                avg_acc += acc
                avg_iou += iou
            avg_acc /= len(selected_clients)
            avg_iou /= len(selected_clients)
            history['acc'].append(avg_acc)
            history['iou'].append(avg_iou)
            print(f"本轮平均准确率: {avg_acc:.4f}, 平均IoU: {avg_iou:.4f}")

        return history


# -------------------- 原有辅助函数保持不变 --------------------
# （UNet、数据集类、评估函数、可视化等均未改动，此处仅保留关键部分，完整代码需包含所有原有函数）

# ... [此处省略原有 UNet, DoubleConv, Down, Up, OutConv, 以及数据集类 ImageMaskDataset, NoisyImageMaskDataset,
#      评估函数 evaluate_model, calculate_accuracy, calculate_iou, 可视化函数等，它们与之前完全相同] ...

def get_transforms(image_size=(256, 256)):
    # 图像变换
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 掩码变换（不需要归一化）
    mask_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])
    
    return image_transform, mask_transform

def reset_model_parameters(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def get_device(gpu_id=0):
    if gpu_id == -1:
        device = torch.device('cpu')
        print("use cpu")
    elif torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
            print(f"use GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print(f"GPU {gpu_id} unavailable use GPU 0")
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        print("CUDA unavailable use CPU compute")
    return device

def create_federated_dataloaders(image_folder, mask_folder, num_clients=5, batch_size=4,
                                 image_size=(256, 256), train_ratio=0.8,
                                 val_ratio=0.1, shuffle=True, noise_ratio=0.0):
    """创建联邦学习客户端的数据加载器（与原来相同）"""
    image_transform, mask_transform = get_transforms(image_size)  # 需要修改get_transforms返回两个transform
    # 原函数中get_transforms返回两个，这里需对应调整
    # 为简化，直接调用原函数
    # 注意：原get_transforms返回两个，此处假设已定义
    # 重新实现
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])
    # 创建带噪声的训练数据集
    full_train_dataset = NoisyImageMaskDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        noise_ratio=noise_ratio,
        transform=image_transform,
        target_transform=mask_transform
    )
    full_val_dataset = ImageMaskDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        transform=image_transform,
        target_transform=mask_transform
    )

    total_size = len(full_train_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    indices = list(range(total_size))
    if shuffle:
        random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    full_train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
    full_val_subset = torch.utils.data.Subset(full_val_dataset, val_indices)

    # 分割客户端数据
    def split_dataset(dataset, num_clients):
        total = len(dataset)
        size_per_client = total // num_clients
        subsets = []
        for i in range(num_clients):
            start = i * size_per_client
            end = (i + 1) * size_per_client if i < num_clients - 1 else total
            subsets.append(torch.utils.data.Subset(dataset, list(range(start, end))))
        return subsets

    client_train_subsets = split_dataset(full_train_subset, num_clients)
    client_val_subsets = split_dataset(full_val_subset, num_clients)

    client_loaders = []
    for i in range(num_clients):
        train_loader = DataLoader(client_train_subsets[i], batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(client_val_subsets[i], batch_size=batch_size, shuffle=False)
        client_loaders.append({'train': train_loader, 'val': val_loader})

    test_loader = DataLoader(torch.utils.data.Subset(full_val_dataset, test_indices),
                             batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader

def create_public_loader(image_folder, batch_size=32, image_size=(256,256)):
    """创建公共数据加载器（无标签）"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = PublicDataset(image_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    gpu_id = 1
    device = get_device(gpu_id)
    num_clients = 8

    image_folder = "D:\\Fundus-doFE\\Fundus\\Domain1\\train\\image"
    mask_folder = "D:\\Fundus-doFE\\Fundus\\Domain1\\train\\mask"
    noise_rates = [0.0, 0.2, 0.4, 0.6, 0.8]

    # 创建公共数据加载器（使用同一图像文件夹，但假设无标签）
    public_loader = create_public_loader(image_folder, batch_size=32)

    model_save_dir = "saved_models_for_rhfl"
    os.makedirs(model_save_dir, exist_ok=True)

    for noise_rate in noise_rates:
        print(f"\n========== 训练噪声率为 {noise_rate} 的RHFL模型 ==========")

        # 创建联邦学习客户端数据加载器
        client_loaders, test_loader = create_federated_dataloaders(
            image_folder=image_folder,
            mask_folder=mask_folder,
            num_clients=num_clients,
            batch_size=6,
            image_size=(256, 256),
            noise_ratio=noise_rate
        )

        # 创建客户端实例（每个客户端独立模型）
        clients = []
        for i in range(num_clients):
            model = UNet(n_channels=3, n_classes=3)
            reset_model_parameters(model)
            client = Client(
                client_id=i,
                model=model,
                train_loader=client_loaders[i]['train'],
                val_loader=client_loaders[i]['val'],
                public_loader=public_loader,
                device=device
            )
            clients.append(client)

        # 创建RHFL服务器
        server = RHFLServer(clients, public_loader, device, eta=0.5)

        # 执行训练
        history = server.federated_train(
            rounds=20,
            local_epochs=5,
            client_fraction=0.8,
            learning_rate=0.001
        )

        # 评估所有客户端在测试集上的平均性能
        acc_list, iou_list = [], []
        for client in clients:
            acc, iou = evaluate_model(client.model, test_loader, device)
            acc_list.append(acc)
            iou_list.append(iou)
        mean_acc = np.mean(acc_list)
        mean_iou = np.mean(iou_list)
        print(f"噪声率 {noise_rate} 所有客户端测试平均 - 准确率: {mean_acc:.4f}, IoU: {mean_iou:.4f}")

        # 保存每个客户端的模型
        for i, client in enumerate(clients):
            model_path = os.path.join(model_save_dir, f"client_{i}_noise_{noise_rate}.pth")
            torch.save(client.model.state_dict(), model_path)
        print(f"模型已保存至 {model_save_dir}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()