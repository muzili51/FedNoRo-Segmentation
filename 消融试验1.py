# train_FedNoRo.py
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
import numpy as np
from torchvision.utils import make_grid
from utils.options_for_消融试验1 import args_parser
args = args_parser()
import random
from dataset.all_datasets import (
    ImageMaskDataset,
    NoisyImageMaskDataset,
    create_federated_dataloaders,
    add_morphological_noise_to_mask
)
from model.all_models import get_model
from utils.Server_for_消融试验1 import FedAvgServer
from utils.Client import Client, evaluate_model, calculate_accuracy, calculate_iou
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

def save_model(model, save_path, epoch=None, optimizer=None, loss=None):
    """
    保存模型
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"模型已保存至: {save_path}")


if __name__ == "__main__":
    gpu_id = args.gpu
    device = get_device(gpu_id)

    # 创建全局模型
    global_model = get_model(
        model_name=args.model,  # 从options中获取模型名称
    )
    image_folder = "D:\\Fundus-doFE\\Fundus\\Domain1\\train\\image"
    mask_folder = "D:\\Fundus-doFE\\Fundus\\Domain1\\train\\mask"

    # 创建联邦学习数据加载器
    num_clients = args.n_clients
    client_loaders, test_loader = create_federated_dataloaders(
        image_folder=image_folder,
        mask_folder=mask_folder,
        num_clients=num_clients,
        batch_size=args.batch_size,
        image_size=(256, 256),
        noise_ratio=0.8,
        noise_client_ratio=0.6  # 使用参数指定的噪声客户端比例
    )

    # 创建客户端实例
    clients = []
    true_noise_clients = []
    true_clean_clients = []
    for i in range(num_clients):
        # 为每个客户端创建独立的模型实例
        client_model = get_model(
            model_name=args.model,
        )
        client = Client(
            client_id=i,
            model=client_model,
            train_loader=client_loaders[i]['train'],
            val_loader=client_loaders[i]['val'],
            device=device,
            has_noise=client_loaders[i]['has_noise']
        )
        clients.append(client)
    
    for i in range(num_clients):
        if client_loaders[i]['has_noise']:
            true_noise_clients.append(i)
        else:
            true_clean_clients.append(i)
    print(f"有噪声客户端ID：{true_noise_clients}", f"无噪声客户端ID：{true_clean_clients}")
    
    # 创建联邦学习服务器
    server = FedAvgServer(global_model, clients, device)
    #-------rounds 1--------#
    trained_global_model, history = server.federated_train(
        rounds=30,
        local_epochs=5,
        learning_rate=args.base_lr,
        stage1_rounds=30
    )
    print(f"检测到的噪声客户端: {server.noisy_clients}")
    print(f"检测到的干净客户端: {server.clean_clients}")
    print(f"所有噪声客户端识别正确率为：{server.overall_noise_clients_detect_performance}")
    # 绘制折线图
    plt.figure(figsize=(10, 6))

    x_indices = range(len(server.overall_noise_clients_detect_performance))
    line2, = plt.plot(x_indices, server.overall_recall, marker='s', linestyle='--', color='r', label='Overall Recall')
    line3, = plt.plot(x_indices, server.overall_precision, marker='^', linestyle='-.', color='g', label='Overall Precision')
    line4, = plt.plot(x_indices, server.overall_match_ratio, marker='d', linestyle=':', color='m', label='Overall Match Ratio')

    plt.title('Federated Learning Noise Clients Detect Performance Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Value')
    plt.grid(True)

    # 添加图例
    plt.legend()

    # 保存图像到本地
    plt.savefig('消融试验1.png')

    # 显示图像（可选）
    plt.show()
0