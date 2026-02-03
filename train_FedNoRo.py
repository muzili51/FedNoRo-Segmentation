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
from utils.options import args_parser
args = args_parser()
import random
from dataset.all_datasets import (
    ImageMaskDataset,
    NoisyImageMaskDataset,
    create_federated_dataloaders,
    add_morphological_noise_to_mask
)
from model.all_models import get_model
from model.dual_network import DualNetwork, get_dual_model
from utils.Client import Client, evaluate_model, calculate_accuracy, calculate_iou, DualNetworkClient
from utils.Server import FedAvgServer


# 其他函数保持不变
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


def plot_training_history(train_losses, val_dices, val_ious):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, val_dices, 'g-', label='Validation Dice')
    ax2.plot(epochs, val_ious, 'r-', label='Validation IoU')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def save_sample_predictions(model, dataloader, device, save_dir, num_samples=20):  # 修改默认值为20
    import os
    from PIL import Image
    import torch.nn.functional as F

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

                # 获取输入图像
                img = images[i].cpu() * std + mean
                img = torch.clamp(img, 0, 1)
                img = (img * 255).byte()
                img_pil = transforms.ToPILImage()(img)

                # 获取真实标签
                mask_np = masks[i].cpu().numpy().astype(np.uint8)
                # 将标签转换为彩色图像以便可视化
                class_colors = np.array([
                    [0, 0, 0],      # 背景
                    [255, 0, 0],    # 类别1 (红色)
                    [0, 255, 0]     # 类别2 (绿色)
                ], dtype=np.uint8)
                mask_colored = class_colors[mask_np]
                mask_pil = Image.fromarray(mask_colored, mode='RGB')

                # 获取预测结果
                pred_np = predictions[i].cpu().numpy().astype(np.uint8)
                pred_colored = class_colors[pred_np]
                pred_pil = Image.fromarray(pred_colored, mode='RGB')

                # 创建拼接图像 (宽度为3倍原图宽，高度为原图高)
                original_width, original_height = img_pil.size
                combined_img = Image.new('RGB', (original_width * 3, original_height))
                combined_img.paste(img_pil, (0, 0))
                combined_img.paste(mask_pil, (original_width, 0))
                combined_img.paste(pred_pil, (original_width * 2, 0))

                # 保存拼接图像
                combined_img.save(os.path.join(save_dir, f'combined_{img_idx}.png'))

                if img_idx >= num_samples - 1:
                    return


def dual_network_train_local(client, epochs=5, learning_rate=0.001):
    """
    双网络客户端的本地训练函数
    """
    client.model1.to(client.device)
    client.model2.to(client.device)
    client.model1.train()
    client.model2.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(client.model1.parameters()) + list(client.model2.parameters()), 
        lr=learning_rate
    )

    running_loss = 0.0
    for epoch in range(epochs):
        for batch_idx, (images, masks) in enumerate(client.train_loader):
            images = images.to(client.device)
            masks = masks.to(client.device)

            optimizer.zero_grad()
            
            # 获取两个网络的输出
            output1 = client.model1(images)
            output2 = client.model2(images)
            
            # 确保标签值在有效范围内
            if masks.min() < 0 or masks.max() >= client.model1.n_classes:
                masks = torch.clamp(masks, 0, client.model1.n_classes - 1)
            
            # 计算两个网络的损失
            loss1 = criterion(output1, masks)
            loss2 = criterion(output2, masks)
            
            # 总损失
            loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    avg_loss = running_loss / len(client.train_loader)
    return avg_loss


def advanced_federated_training_with_dual_networks(args, server, clients):
    # -----rounds 2------#
    device = server.device
    global_model = server.global_model
    # 存储训练历史
    global_train_losses = []
    global_val_dices = []
    global_val_ious = []
    all_round_avg_losses = []
    # 第二阶段：噪声客户端使用双网络，干净客户端使用单网络
    print("=== 第二阶段训练：差异化网络训练 ===")
    detected_noisy = server.noisy_clients
    for round_num in range(args.s1, args.rounds):

        print(f"\n--- 第二阶段第 {round_num - args.s1 + 1}/{args.rounds - args.s1} 轮 (总轮次: {round_num + 1}) ---")

        # 为噪声客户端转换为双网络训练
        updated_clients = []
        for i, client in enumerate(clients):
            if i in detected_noisy:
                # 将噪声客户端转换为双网络
                if not hasattr(client, 'model1'):  # 如果还不是双网络
                    model1 = get_model(args.model, args.pretrained)
                    model2 = get_model(args.model, args.pretrained)
                    dual_client = DualNetworkClient(
                        client_id=client.client_id,
                        model1=model1,
                        model2=model2,
                        train_loader=client.train_loader,
                        val_loader=client.val_loader,
                        device=device,
                        has_noise=True
                    )
                    updated_clients.append(dual_client)
                else:
                    updated_clients.append(client)
            else:
                # 干净客户端继续使用单网络
                updated_clients.append(client)
        
        clients = updated_clients

        # 收集客户端数据大小
        client_data_sizes = []
        client_models = []
        round_avg_losses = []

        # 将全局模型分发给所有客户端
        for client in clients:
            if client.client_id in detected_noisy and hasattr(client, 'model1'):
                # 双网络客户端，更新两个子网络
                client.model1.load_state_dict(global_model.state_dict())
                client.model2.load_state_dict(global_model.state_dict())
                client_data_sizes.append(len(client.train_loader.dataset))
            else:
                # 单网络客户端
                client.model.load_state_dict(global_model.state_dict())
                client_data_sizes.append(len(client.train_loader.dataset))

        # 客户端依次本地训练
        for client in clients:
            print(f"客户端 {client.client_id} 开始本地训练...")
            
            if client.client_id in detected_noisy and hasattr(client, 'model1'):
                # 双网络训练
                avg_loss = dual_network_train_local(client, args.local_ep, args.base_lr)
                # 使用第一个网络作为代表参与聚合
                client_models.append(client.model1)
            else:
                # 单网络训练
                avg_loss = client.train_local(epochs=args.local_ep, learning_rate=args.base_lr)
                client_models.append(client.model)
            
            round_avg_losses.append(avg_loss)
            print(f"客户端 {client.client_id} 本地训练完成，平均Loss: {avg_loss:.4f}")

            # 评估客户端模型
            dice, iou = client.evaluate_local()
            print(f"客户端 {client.client_id} 本地评估 - Dice: {dice:.4f}, IoU: {iou:.4f}")

        # 计算并输出当前轮次的平均loss
        overall_avg_loss = sum(round_avg_losses) / len(round_avg_losses)
        all_round_avg_losses.append(overall_avg_loss)
        print(f"第 {round_num + 1} 轮整体平均Loss: {overall_avg_loss:.4f}")

        # 使用 DaAgg 聚合客户端模型（替代 FedAvg）
        print("执行 DaAgg 动态聚合...")
        # 评估每个客户端模型的性能（使用验证集）
        client_performances = server.evaluate_client_performance(client_models, clients[0].val_loader)
        
        # 使用 DaAgg 进行动态聚合
        global_model = server.aggregate_weights_daagg(client_models, client_data_sizes, client_performances)

        # 评估全局模型
        if len(clients) > 0:
            temp_test_loader = clients[0].val_loader
            global_dice, global_iou = evaluate_model(global_model, temp_test_loader, device)

            global_val_dices.append(global_dice)
            global_val_ious.append(global_iou)

            print(f"全局模型评估 - Dice: {global_dice:.4f}, IoU: {global_iou:.4f}")

        print(f"第 {round_num + 1} 轮训练完成")

    return global_model, (global_val_dices, global_val_ious, all_round_avg_losses)

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
        noise_client_ratio=args.noise_client_ratio  # 使用参数指定的噪声客户端比例
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
        rounds=args.s1,
        local_epochs=5,
        learning_rate=args.base_lr,
        stage1_rounds=args.s1
    )
    print(f"检测到的噪声客户端: {server.noisy_clients}")
    print(f"检测到的干净客户端: {server.clean_clients}")
    #-------rounds 2--------#
    # 第二阶段训练：使用双网络训练噪声客户端
    remaining_rounds = args.rounds - args.s1
    if remaining_rounds > 0:
        trained_global_model, history = advanced_federated_training_with_dual_networks(
            args, server, clients
        )
    
    # 评估最终全局模型
    final_dice, final_iou = evaluate_model(trained_global_model, test_loader, device)
    print(f"最终全局模型 - 测试Dice: {final_dice:.4f}, 测试IoU: {final_iou:.4f}")

    # 保存训练完成的模型
    model_save_path = f"saved_models/{args.model}_{args.noise_client_ratio}_FedNoRo_round_{args.rounds}.pth"
    save_model(trained_global_model, model_save_path, epoch=args.rounds)
    print(f"模型已保存至: {model_save_path}")

    print("可视化全局模型预测结果...")
    visualize_single_prediction(trained_global_model, test_loader, device, num_samples=4)

    print("保存全局模型预测结果...")
    save_sample_predictions(trained_global_model, test_loader, device, f"fedavg_global_predictions_{args.model}_{args.noise_client_ratio}_{args.rounds}", num_samples=20)  # 修改为20张

    # 输出检测结果总结
    print(f"\n检测结果总结:")
    print(f"实际噪声客户端: {true_noise_clients}")
    print(f"实际干净客户端: {true_clean_clients}")
    print(f"检测到的噪声客户端: {server.noisy_clients}")
    print(f"检测到的干净客户端: {server.clean_clients}")
    
    if true_noise_clients:
        detection_accuracy = len(set(server.noisy_clients) & set(true_noise_clients)) / len(true_noise_clients)
        print(f"噪声客户端检测准确率: {detection_accuracy:.2f}")
    
    # 可视化训练历史
    if history and len(history[0]) > 0:
        epochs = range(1, len(history[0]) + 1)

        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, history[2], 'b-', label='Global Training Loss')  # loss历史
        plt.title('Global Model Training Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, history[0], 'g-', label='Global Validation Dice')
        plt.title('Global Model Validation Dice')
        plt.xlabel('Round')
        plt.ylabel('Dice')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, history[1], 'r-', label='Global Validation IoU')
        plt.title('Global Model Validation IoU')
        plt.xlabel('Round')
        plt.ylabel('IoU')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()