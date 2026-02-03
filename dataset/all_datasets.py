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
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random


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
                 transform=None, target_transform=None, client_id=None):
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
        self.client_id = client_id
        self.noise_ratio = noise_ratio
        self.samples_with_noise = 0

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
            self.samples_with_noise += 1
            if self.client_id is not None:
                print(f"客户端 {self.client_id}: 为样本 {idx} 添加了噪声")

        return image, mask


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
                                 val_ratio=0.1, shuffle=True, noise_ratio=0.0,
                                 noise_client_ratio=0.0):
    """
    为联邦学习创建多个客户端的数据加载器
    noise_client_ratio: 添加噪声的客户端比例
    """
    image_transform, mask_transform = get_transforms(image_size)

    # 创建完整的无噪声训练数据集
    full_train_dataset = ImageMaskDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
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

    # 确定哪些客户端添加噪声
    num_noisy_clients = int(num_clients * noise_client_ratio)
    noisy_client_indices = random.sample(range(num_clients), num_noisy_clients)
    print(f"添加噪声的客户端索引: {noisy_client_indices} (总共{num_clients}个客户端，{noise_client_ratio*100}%添加噪声)")

    # 为所有客户端创建数据加载器
    client_loaders = []
    for i in range(num_clients):
        if i in noisy_client_indices:
            # 为该客户端创建带噪声的训练数据集
            noisy_train_dataset = NoisyImageMaskDataset(
                image_folder=image_folder,
                mask_folder=mask_folder,
                noise_ratio=noise_ratio,
                transform=image_transform,
                target_transform=mask_transform
            )
            # 重新分割这个客户端的数据
            client_train_dataset = torch.utils.data.Subset(noisy_train_dataset,
                                                         client_train_datasets[i].indices)
        else:
            # 为该客户端创建无噪声的训练数据集
            client_train_dataset = client_train_datasets[i]

        train_loader = torch.utils.data.DataLoader(
            client_train_dataset, batch_size=batch_size, shuffle=shuffle
        )

        val_loader = torch.utils.data.DataLoader(
            client_val_datasets[i], batch_size=batch_size, shuffle=False
        )

        client_loaders.append({
            'train': train_loader,
            'val': val_loader,
            'has_noise': i in noisy_client_indices  # 标记是否包含噪声
        })

    # 创建全局测试加载器
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_val_dataset, test_indices),
        batch_size=batch_size, shuffle=False
    )

    return client_loaders, test_loader