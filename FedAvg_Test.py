import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import copy
from collections import OrderedDict
import math

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

import random


class Client:
    def __init__(self, client_id, model, train_loader, val_loader, device):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_local(self, epochs=5, learning_rate=0.001):
        """
        在客户端本地训练模型
        """
        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        running_loss = 0.0
        for epoch in range(epochs):
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)

                # 确保标签值在有效范围内
                if masks.min() < 0 or masks.max() >= self.model.n_classes:
                    masks = torch.clamp(masks, 0, self.model.n_classes - 1)

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss


    def evaluate_local(self):
        """
        在客户端本地评估模型
        """
        return evaluate_model(self.model, self.val_loader, self.device)


class FedAvgServer:
    def __init__(self, global_model, clients, device):
        self.global_model = global_model
        self.clients = clients
        self.device = device

    def aggregate_weights_fedavg(self, client_models, client_data_sizes):
        """
        FedAvg 权重聚合算法
        w_avg = sum(n_k/n * w_k) for k=1 to K
        其中 n_k 是客户端 k 的数据量，n 是总数据量
        """
        # 获取总数据量
        total_data_size = sum(client_data_sizes)

        # 初始化聚合权重
        aggregated_state_dict = OrderedDict()

        # 获取第一个模型的参数作为参考
        first_model_params = list(client_models[0].state_dict().values())

        # 遍历所有参数
        for param_idx, param_name in enumerate(self.global_model.state_dict().keys()):
            # 计算加权平均
            weighted_param = None

            for client_idx, client_model in enumerate(client_models):
                client_param = list(client_model.state_dict().values())[param_idx]
                weight = client_data_sizes[client_idx] / total_data_size

                if weighted_param is None:
                    weighted_param = weight * client_param.clone()
                else:
                    weighted_param += weight * client_param.clone()

            aggregated_state_dict[param_name] = weighted_param

        # 更新全局模型参数
        self.global_model.load_state_dict(aggregated_state_dict)

        return self.global_model

    def select_clients(self, fraction=0.8):
        """
        随机选择一部分客户端参与训练
        """
        num_clients = len(self.clients)
        num_selected = max(1, int(fraction * num_clients))
        selected_indices = np.random.choice(len(self.clients), num_selected, replace=False)
        selected_clients = [self.clients[i] for i in selected_indices]

        return selected_clients

    def federated_train(self, rounds=50, local_epochs=5, client_fraction=0.8,
                        learning_rate=0.001):
        """
        联邦学习训练主循环
        """
        print(f"开始联邦学习训练，共 {rounds} 轮")

        # 存储训练历史
        global_train_losses = []
        global_val_accuracies = []
        global_val_ious = []

        for round_num in range(rounds):
            print(f"\n=== 联邦训练第 {round_num + 1} 轮 ===")

            # 选择参与本轮训练的客户端
            selected_clients = self.select_clients(client_fraction)
            print(f"选择 {len(selected_clients)} 个客户端参与训练")

            # 收集客户端数据大小
            client_data_sizes = []
            client_models = []

            # 将全局模型分发给选中的客户端
            for client in selected_clients:
                # 复制全局模型到客户端
                client.model.load_state_dict(self.global_model.state_dict())

                # 记录客户端数据大小
                client_data_sizes.append(len(client.train_loader.dataset))
                print(f"客户端 {client.client_id} 数据量: {len(client.train_loader.dataset)}")

            # 客户端本地训练
            for client in selected_clients:
                print(f"客户端 {client.client_id} 开始本地训练...")
                client.train_local(epochs=local_epochs, learning_rate=learning_rate)

                # 收集训练后的模型
                client_models.append(copy.deepcopy(client.model))

                # 评估客户端模型
                acc, iou = client.evaluate_local()
                print(f"客户端 {client.client_id} 本地评估 - 准确率: {acc:.4f}, IoU: {iou:.4f}")

            # 使用 FedAvg 聚合客户端模型
            print("执行 FedAvg 聚合...")
            self.global_model = self.aggregate_weights_fedavg(client_models, client_data_sizes)

            # 评估全局模型
            # 创建临时测试加载器（这里使用第一个客户端的验证集作为全局评估）
            if len(selected_clients) > 0:
                temp_test_loader = selected_clients[0].val_loader
                global_acc, global_iou = evaluate_model(self.global_model, temp_test_loader, self.device)

                global_val_accuracies.append(global_acc)
                global_val_ious.append(global_iou)

                print(f"全局模型评估 - 准确率: {global_acc:.4f}, IoU: {global_iou:.4f}")

            print(f"第 {round_num + 1} 轮训练完成")

        return self.global_model, (global_val_accuracies, global_val_ious)


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


def train_model_with_history(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, gpu_id=0):
    device = get_device(gpu_id)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []
    val_ious = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if masks.min() < 0 or masks.max() >= model.n_classes:
                masks = torch.clamp(masks, 0, model.n_classes - 1)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_accuracy, val_iou = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        val_ious.append(val_iou)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Loss: {avg_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, Val IoU: {val_iou:.4f}')

    plot_training_history(train_losses, val_accuracies, val_ious)

    return model, (train_losses, val_accuracies, val_ious)


def get_transforms(image_size=(256, 256)):
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

    return image_transform, mask_transform


def create_dataloaders_with_morphological_noise(image_folder, mask_folder, batch_size=4,
                                                image_size=(256, 256), train_ratio=0.8,
                                                val_ratio=0.1, shuffle=True, noise_ratio=0.0):
    """
    创建带有指定比例腐蚀/膨胀噪声的数据加载器

    Args:
        noise_ratio: 添加腐蚀/膨胀噪声的掩码比例 (0.0-1.0)
    """
    image_transform, mask_transform = get_transforms(image_size)

    # 训练集添加腐蚀/膨胀噪声
    train_dataset = NoisyImageMaskDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        noise_ratio=noise_ratio,  # 训练集添加噪声
        transform=image_transform,
        target_transform=mask_transform
    )

    # 创建完整的数据集用于分割（验证和测试集不添加噪声）
    full_dataset = ImageMaskDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        transform=image_transform,
        target_transform=mask_transform
    )

    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # 分割数据集索引
    indices = list(range(total_size))
    if shuffle:
        random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建验证和测试数据集（不添加噪声）
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def create_dataloaders(image_folder, mask_folder, batch_size=4,
                       image_size=(256, 256), train_ratio=0.8,
                       val_ratio=0.1, shuffle=True):
    image_transform, mask_transform = get_transforms(image_size)

    dataset = ImageMaskDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        transform=image_transform,
        target_transform=mask_transform
    )

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


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


def split_data_for_federated_learning(dataset, num_clients, iid_distribution=False):
    """
    将数据集拆分为多个子集供不同客户端使用
    """
    total_size = len(dataset)
    samples_per_client = total_size // num_clients

    client_datasets = []

    if iid_distribution:
        # IID分布：随机分配数据
        indices = list(range(total_size))
        np.random.shuffle(indices)

        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_size
            client_indices = indices[start_idx:end_idx]

            client_dataset = torch.utils.data.Subset(dataset, client_indices)
            client_datasets.append(client_dataset)
    else:
        # Non-IID分布：按类别分配数据
        # 这里简单实现按顺序分配
        indices = list(range(total_size))

        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_size
            client_indices = indices[start_idx:end_idx]

            client_dataset = torch.utils.data.Subset(dataset, client_indices)
            client_datasets.append(client_dataset)

    return client_datasets


def create_federated_dataloaders(image_folder, mask_folder, num_clients=5, batch_size=4,
                                 image_size=(256, 256), train_ratio=0.8,
                                 val_ratio=0.1, shuffle=True, noise_ratio=0.0):
    """
    为联邦学习创建多个客户端的数据加载器
    """
    image_transform, mask_transform = get_transforms(image_size)

    # 创建带噪声的训练数据集
    full_train_dataset = NoisyImageMaskDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        noise_ratio=noise_ratio,
        transform=image_transform,
        target_transform=mask_transform
    )

    # 创建完整的无噪声验证数据集
    full_val_dataset = ImageMaskDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        transform=image_transform,
        target_transform=mask_transform
    )

    # 分割数据集
    total_size = len(full_train_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # 分割训练和验证索引
    indices = list(range(total_size))
    if shuffle:
        random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建训练和验证子集
    full_train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
    full_val_subset = torch.utils.data.Subset(full_val_dataset, val_indices)

    # 为每个客户端分割数据
    client_train_datasets = split_data_for_federated_learning(full_train_subset, num_clients)
    client_val_datasets = split_data_for_federated_learning(full_val_subset, num_clients)

    # 为所有客户端创建数据加载器
    client_loaders = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(
            client_train_datasets[i], batch_size=batch_size, shuffle=shuffle
        )

        val_loader = torch.utils.data.DataLoader(
            client_val_datasets[i], batch_size=batch_size, shuffle=False
        )

        client_loaders.append({
            'train': train_loader,
            'val': val_loader
        })

    # 创建全局测试加载器
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_val_dataset, test_indices),
        batch_size=batch_size, shuffle=False
    )

    return client_loaders, test_loader

def visualize_batch(images, masks, predictions=None, num_samples=4, class_colors=None):
    if class_colors is None:
        class_colors = np.array([
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0]
        ], dtype=np.uint8)

    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(3 if predictions is not None else 2, num_samples,
                             figsize=(num_samples * 4, 8 if predictions is not None else 6))

    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_samples):
        img = images[i].cpu().permute(1, 2, 0)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        if num_samples > 1:
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Input Image {i + 1}')
            axes[0, i].axis('off')
        else:
            axes[0].imshow(img)
            axes[0].set_title(f'Input Image {i + 1}')
            axes[0].axis('off')

        mask = masks[i].cpu().numpy()
        mask_colored = class_colors[mask]

        if num_samples > 1:
            axes[1, i].imshow(mask_colored)
            axes[1, i].set_title(f'Ground Truth {i + 1}')
            axes[1, i].axis('off')
        else:
            axes[1].imshow(mask_colored)
            axes[1].set_title(f'Ground Truth {i + 1}')
            axes[1].axis('off')

        if predictions is not None:
            pred = predictions[i].cpu().numpy()
            pred_colored = class_colors[pred]

            if num_samples > 1:
                axes[2, i].imshow(pred_colored)
                axes[2, i].set_title(f'Prediction {i + 1}')
                axes[2, i].axis('off')
            else:
                axes[2].imshow(pred_colored)
                axes[2].set_title(f'Prediction {i + 1}')
                axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_single_prediction(model, dataloader, device, num_samples=4, class_colors=None):
    model.eval()

    with torch.no_grad():
        images, masks = next(iter(dataloader))
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        visualize_batch(images, masks, predictions, num_samples, class_colors)


def plot_training_history(train_losses, val_accuracies, val_ious):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.plot(epochs, val_ious, 'r-', label='Validation IoU')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def save_sample_predictions(model, dataloader, device, save_dir, num_samples=5):
    import os
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            for i in range(min(num_samples, len(images))):
                img_idx = batch_idx * len(images) + i

                img = images[i].cpu() * std + mean
                img = torch.clamp(img, 0, 1)
                img = (img * 255).byte()
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(os.path.join(save_dir, f'input_{img_idx}.png'))

                mask_pil = Image.fromarray(masks[i].cpu().numpy().astype(np.uint8), mode='L')
                mask_pil.save(os.path.join(save_dir, f'ground_truth_{img_idx}.png'))

                pred_pil = Image.fromarray(predictions[i].cpu().numpy().astype(np.uint8), mode='L')
                pred_pil.save(os.path.join(save_dir, f'prediction_{img_idx}.png'))

                if img_idx >= num_samples - 1:
                    return

def reset_model_parameters(model):
    """
    重置模型的所有可学习参数为随机初始化值
    """
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

if __name__ == "__main__":
    gpu_id = 1
    device = get_device(gpu_id)
    num_clients = 8
    # 创建全局模型
    global_model = UNet(n_channels=3, n_classes=3)

    image_folder = "D:\\Fundus-doFE\\Fundus\\Domain1\\train\\image"
    mask_folder = "D:\\Fundus-doFE\\Fundus\\Domain1\\train\\mask"

    noise_rates = [0.6, 0.8]

    # 创建保存模型的目录
    model_save_dir = "saved_models_for_fedavg"
    os.makedirs(model_save_dir, exist_ok=True)

    # 循环遍历不同的噪声率
    for noise_rate in noise_rates:
        print(f"\n开始训练噪声率为 {noise_rate} 的模型...")

        # 创建联邦学习数据加载器
        client_loaders, test_loader = create_federated_dataloaders(
            image_folder=image_folder,
            mask_folder=mask_folder,
            num_clients=num_clients,
            batch_size=8,
            image_size=(256, 256),
            noise_ratio=noise_rate  # 使用当前噪声率
        )

        # 创建客户端实例
        clients = []
        for i in range(num_clients):
            client_model = UNet(n_channels=3, n_classes=3)
            reset_model_parameters(client_model)
            client = Client(
                client_id=i,
                model=client_model,
                train_loader=client_loaders[i]['train'],
                val_loader=client_loaders[i]['val'],
                device=device
            )
            clients.append(client)

        # 创建联邦学习服务器
        server = FedAvgServer(global_model, clients, device)

        # 执行联邦学习训练
        trained_global_model, history = server.federated_train(
            rounds=20,
            local_epochs=5,
            client_fraction=0.8,
            learning_rate=0.001
        )

        # 评估最终全局模型
        final_accuracy, final_iou = evaluate_model(trained_global_model, test_loader, device)
        print(f"噪声率为 {noise_rate} 的模型 - 测试准确率: {final_accuracy:.4f}, 测试IoU: {final_iou:.4f}")

        # 保存模型
        model_save_path = os.path.join(model_save_dir, f"fedavg_model_noise_{noise_rate}.pth")
        torch.save(trained_global_model.state_dict(), model_save_path)
        print(f"模型已保存至: {model_save_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reset_model_parameters(global_model)