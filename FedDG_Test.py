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
from FedAvg_Test import UNet, create_federated_dataloaders, FedAvgServer,get_device,reset_model_parameters

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

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, num_domains):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains)
        )

    def forward(self, x):
        return self.classifier(x)
    
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
    
class FedDGClient(Client):
    def __init__(self, client_id, model, domain_classifier, train_loader, val_loader, device, domain_label):
        super(FedDGClient, self).__init__(client_id, model, train_loader, val_loader, device)
        self.domain_classifier = domain_classifier
        self.domain_label = domain_label  # 当前客户端所属的域标签

    def train_local(self, epochs=5, learning_rate=0.001, alpha=0.1):
        """
        在客户端本地训练模型，同时进行域对抗训练
        """
        self.model.to(self.device)
        self.domain_classifier.to(self.device)
        self.model.train()
        self.domain_classifier.train()

        seg_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.domain_classifier.parameters()), lr=learning_rate)

        running_loss = 0.0
        for epoch in range(epochs):
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                features = self.model.inc(images)  # 提取特征
                outputs = self.model(images)       # 分割输出
                domain_preds = self.domain_classifier(features)  # 域预测

                # 计算损失
                seg_loss = seg_criterion(outputs, masks)
                domain_labels = torch.full((images.size(0),), self.domain_label, dtype=torch.long).to(self.device)
                domain_loss = domain_criterion(domain_preds, domain_labels)

                # 总损失 = 分割损失 + 域对抗损失
                total_loss = seg_loss + alpha * domain_loss
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
    
class FedDGServer(FedAvgServer):
    def aggregate_weights_feddg(self, client_models, client_data_sizes, domain_weights=None):
        """
        FedDG 权重聚合算法：引入域权重以增强泛化能力
        """
        total_data_size = sum(client_data_sizes)
        aggregated_state_dict = OrderedDict()

        for param_name in self.global_model.state_dict().keys():
            weighted_param = None
            for client_idx, client_model in enumerate(client_models):
                client_param = client_model.state_dict()[param_name]
                weight = client_data_sizes[client_idx] / total_data_size
                if domain_weights is not None:
                    weight *= domain_weights[client_idx]  # 引入域权重

                if weighted_param is None:
                    weighted_param = weight * client_param.clone()
                else:
                    weighted_param += weight * client_param.clone()

            aggregated_state_dict[param_name] = weighted_param

        self.global_model.load_state_dict(aggregated_state_dict)
        return self.global_model
    
if __name__ == "__main__":
    gpu_id = 1
    device = get_device(gpu_id)
    num_clients = 8
    num_domains = 3

    # 创建全局模型和域分类器
    global_model = UNet(n_channels=3, n_classes=3)
    domain_classifier = DomainClassifier(input_dim=64, num_domains=num_domains)

    image_folder = "D:\\Fundus-doFE\\Fundus\\Domain1\\train\\image"
    mask_folder = "D:\\Fundus-doFE\\Fundus\\Domain1\\train\\mask"

    noise_rates = [0.0, 0.2, 0.4, 0.6, 0.8]

    # 创建保存模型的目录
    model_save_dir = "saved_models_for_feddg"
    os.makedirs(model_save_dir, exist_ok=True)

    # 循环遍历不同的噪声率
    for noise_rate in noise_rates:
        print(f"\n开始训练噪声率为 {noise_rate} 的 FedDG 模型...")

        # 创建联邦学习数据加载器
        client_loaders, test_loader = create_federated_dataloaders(
            image_folder=image_folder,
            mask_folder=mask_folder,
            num_clients=num_clients,
            batch_size=8,
            image_size=(256, 256),
            noise_ratio=noise_rate
        )

        # 创建客户端实例
        clients = []
        for i in range(num_clients):
            client_model = copy.deepcopy(global_model)
            client_domain_classifier = copy.deepcopy(domain_classifier)
            reset_model_parameters(client_model)
            reset_model_parameters(client_domain_classifier)
            client = Client(
                client_id=i,
                model=client_model,
                # domain_classifier=client_domain_classifier,
                train_loader=client_loaders[i]['train'],
                val_loader=client_loaders[i]['val'],
                device=device,
                # domain_label=i % num_domains
            )
            clients.append(client)

        # 创建 FedDG 服务器
        server = FedAvgServer(global_model, clients, device)

        # 执行 FedDG 联邦学习训练
        trained_global_model, history = server.federated_train(
            rounds=20,
            local_epochs=5,
            client_fraction=0.8,
            learning_rate=0.001,
            # domain_weights=[1.0] * num_clients
        )

        # 评估最终全局模型
        final_accuracy, final_iou = evaluate_model(trained_global_model, test_loader, device)
        print(f"噪声率为 {noise_rate} 的 FedDG 模型 - 测试准确率: {final_accuracy:.4f}, 测试IoU: {final_iou:.4f}")

        # 保存模型
        model_save_path = os.path.join(model_save_dir, f"feddg_model_noise_{noise_rate}.pth")
        torch.save(trained_global_model.state_dict(), model_save_path)
        print(f"模型已保存至: {model_save_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reset_model_parameters(global_model)
        reset_model_parameters(domain_classifier)
        