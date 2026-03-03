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
from FedAvg_Test import reset_model_parameters

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


def train_model_with_history(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, gpu_id=0):
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


if __name__ == "__main__":
    gpu_id = 0
    device = get_device(gpu_id)
    model = UNet(n_channels=3, n_classes=3)
    model_save_dir = "saved_models_for_unet"
    os.makedirs(model_save_dir, exist_ok=True)
    noise_rates = [0.0, 0.2, 0.4, 0.6, 0.8]

    image_folder = "D:\\kvasir-seg\\train\\images"
    mask_folder = "D:\\kvasir-seg\\train\\masks"
    for noise_rate in noise_rates:
        print(f"\n开始训练噪声率为 {noise_rate} 的模型...")
        # 创建带有30%腐蚀/膨胀噪声的数据加载器（仅训练集添加噪声）
        dataloaders = create_dataloaders_with_morphological_noise(
            image_folder=image_folder,
            mask_folder=mask_folder,
            batch_size=8,
            image_size=(256, 256),
            noise_ratio=noise_rate
        )

        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']

        trained_model, history = train_model_with_history(model, train_loader, val_loader, gpu_id=gpu_id)

        accuracy, iou = evaluate_model(trained_model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.4f}, Test IoU: {iou:.4f}")
        accuracy, iou = evaluate_model(trained_model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.4f}, Test IoU: {iou:.4f}")
        model_save_path = os.path.join(model_save_dir, f"Unet_kvasir-seg_model_noise_{noise_rate}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"模型已保存至: {model_save_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reset_model_parameters(model)