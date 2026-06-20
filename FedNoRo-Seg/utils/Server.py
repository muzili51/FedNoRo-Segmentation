# utils/Server.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from .Client import evaluate_model
import numpy as np
from sklearn.mixture import GaussianMixture

class FedAvgServer:
    def __init__(self, global_model, clients, device):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.noisy_clients = []
        self.clean_clients = []

    def aggregate_weights_fedavg(self, client_models, client_data_sizes):
        total_data_size = sum(client_data_sizes)
        aggregated_state_dict = OrderedDict()
        first_model_params = list(client_models[0].state_dict().values())
        for param_idx, param_name in enumerate(self.global_model.state_dict().keys()):
            weighted_param = None

            for client_idx, client_model in enumerate(client_models):
                client_param = list(client_model.state_dict().values())[param_idx]
                weight = client_data_sizes[client_idx] / total_data_size

                if weighted_param is None:
                    weighted_param = weight * client_param.clone()
                else:
                    weighted_param += weight * client_param.clone()

            aggregated_state_dict[param_name] = weighted_param
        self.global_model.load_state_dict(aggregated_state_dict)

        return self.global_model
    
    def evaluate_client_performance(self, client_models, validation_loader):
        performances = []
        criterion = nn.CrossEntropyLoss()
        
        original_model = copy.deepcopy(self.global_model)
        
        for i, client_model in enumerate(client_models):
            client_model.eval()
            client_model.to(self.device)
            
            total_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for images, masks in validation_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = client_model(images)
                    if hasattr(client_model, 'n_classes'):
                        if masks.min() < 0 or masks.max() >= client_model.n_classes:
                            masks = torch.clamp(masks, 0, client_model.n_classes - 1)
                    
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            performances.append(avg_loss)
            
            print(f"客户端 {i} 验证损失: {avg_loss:.4f}")
        self.global_model = original_model
        
        return performances


    def aggregate_weights_daagg(self, client_models, client_data_sizes, client_losses):
        client_losses = np.array(client_losses)
        for client_idx, client_model in enumerate(client_models):
            if client_idx in self.noisy_clients:
                client_losses[client_idx] *= 10
        min_loss = np.min(client_losses)
        max_loss = np.max(client_losses)
        
        if max_loss == min_loss:
            client_weights = np.ones(len(client_losses)) / len(client_losses)
        else:
            normalized_losses = (client_losses - min_loss) / (max_loss - min_loss)
            performance_scores = 1 / (normalized_losses + 1e-8)
            client_weights = performance_scores / np.sum(performance_scores)
        
        print(f"客户端损失: {client_losses}")
        print(f"客户端权重: {client_weights}")
        total_data_size = sum(client_data_sizes)
        aggregated_state_dict = OrderedDict()
        global_keys = self.global_model.state_dict().keys()
        
        for param_idx, param_name in enumerate(global_keys):
            weighted_param = None
            
            for client_idx, client_model in enumerate(client_models):
                client_param = list(client_model.state_dict().values())[param_idx]
                data_weight = client_data_sizes[client_idx] / total_data_size
                combined_weight = client_weights[client_idx]
                
                if weighted_param is None:
                    weighted_param = combined_weight * client_param.clone()
                else:
                    weighted_param += combined_weight * client_param.clone()
            
            aggregated_state_dict[param_name] = weighted_param
        self.global_model.load_state_dict(aggregated_state_dict)
        
        return self.global_model
    
    def evaluate_client_performance(self, client_models, validation_loader):
        performances = []
        criterion = nn.CrossEntropyLoss()
        
        original_model = copy.deepcopy(self.global_model)
        
        for i, client_model in enumerate(client_models):
            client_model.eval()
            client_model.to(self.device)
            
            total_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for images, masks in validation_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = client_model(images)
                    if hasattr(client_model, 'n_classes'):
                        if masks.min() < 0 or masks.max() >= client_model.n_classes:
                            masks = torch.clamp(masks, 0, client_model.n_classes - 1)
                    
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            performances.append(avg_loss)
            
            print(f"客户端 {i} 验证损失: {avg_loss:.4f}")
        self.global_model = original_model
        
        return performances

    def aggregate_weights_daagg(self, client_models, client_data_sizes, client_losses):
        print("执行 DaAgg 动态聚合...")
        client_losses = np.array(client_losses)
        min_loss = np.min(client_losses)
        max_loss = np.max(client_losses)
        
        if max_loss == min_loss:
            client_weights = np.ones(len(client_losses)) / len(client_losses)
        else:
            normalized_losses = (client_losses - min_loss) / (max_loss - min_loss + 1e-8)  # 加小常数避免除零
            performance_scores = 1 / (normalized_losses + 1e-8)
            client_weights = performance_scores / np.sum(performance_scores)
        
        print(f"客户端损失: {client_losses}")
        print(f"客户端权重: {client_weights}")

        total_data_size = sum(client_data_sizes)
        aggregated_state_dict = OrderedDict()
        global_keys = self.global_model.state_dict().keys()
        
        for param_idx, param_name in enumerate(global_keys):
            weighted_param = None
            
            for client_idx, client_model in enumerate(client_models):
                client_param = list(client_model.state_dict().values())[param_idx]
                combined_weight = client_weights[client_idx]
                
                if weighted_param is None:
                    weighted_param = combined_weight * client_param.clone()
                else:
                    weighted_param += combined_weight * client_param.clone()
            
            aggregated_state_dict[param_name] = weighted_param
        self.global_model.load_state_dict(aggregated_state_dict)
        
        return self.global_model
    def evaluate_client_performance(self, client_models, validation_loader):
        performances = []
        criterion = nn.CrossEntropyLoss()
        
        original_model = copy.deepcopy(self.global_model)
        
        for i, client_model in enumerate(client_models):
            client_model.eval()
            client_model.to(self.device)
            
            total_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for images, masks in validation_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = client_model(images)
                    if masks.min() < 0 or masks.max() >= client_model.n_classes:
                        masks = torch.clamp(masks, 0, client_model.n_classes - 1)
                    
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            performances.append(avg_loss)
            
            print(f"客户端 {i} 验证损失: {avg_loss:.4f}")
        self.global_model = original_model
        
        return performances

    def detect_noisy_clients_by_loss(self, client_models, client_loaders):
        print("\n=== 检测噪声客户端 ===")  
        self.global_model.eval()
        n_classes = self.global_model.n_classes
        metrics = np.zeros((len(client_models), n_classes))
        num = np.zeros((len(client_models), n_classes))
        
        for client_id, (client_model, client_loader) in enumerate(zip(client_models, client_loaders)):
            client_model.eval()
            client_model.to(self.device)
            criterion = nn.CrossEntropyLoss(reduction='none')
            
            losses_per_class = [[] for _ in range(n_classes)]
            
            with torch.no_grad():
                for images, masks in client_loader['train']:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = client_model(images)
                    if masks.min() < 0 or masks.max() >= n_classes:
                        masks = torch.clamp(masks, 0, n_classes - 1)
                    per_sample_loss = criterion(outputs, masks)
                    for cls in range(n_classes):
                        class_mask = (masks == cls)
                        class_losses = per_sample_loss[class_mask]
                        if class_losses.numel() > 0:
                            losses_per_class[cls].extend(class_losses.cpu().numpy())
            for cls in range(n_classes):
                if len(losses_per_class[cls]) > 0:
                    metrics[client_id, cls] = np.mean(losses_per_class[cls])
                    num[client_id, cls] = len(losses_per_class[cls])
                else:
                    metrics[client_id, cls] = 0
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                if num[i, j] > 0:
                    metrics[i, j] = metrics[i, j] / num[i, j]
                else:
                    if np.any(metrics[:, j] > 0):
                        metrics[i, j] = np.min(metrics[metrics[:, j] > 0])
                    else:
                        metrics[i, j] = 0
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                if np.isnan(metrics[i, j]):
                    if np.any(~np.isnan(metrics[:, j])):
                        metrics[i, j] = np.nanmin(metrics[:, j])
                    else:
                        metrics[i, j] = 0
        for j in range(metrics.shape[1]):
            min_val = metrics[:, j].min()
            max_val = metrics[:, j].max()
            if max_val != min_val:
                metrics[:, j] = (metrics[:, j] - min_val) / (max_val - min_val)
            else:
                metrics[:, j] = 0
        
        print("客户端指标矩阵:")
        print(metrics)
        vote = []
        for i in range(9):
            gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
            gmm_pred = gmm.predict(metrics)
            means = gmm.means_.sum(axis=1)
            noisy_cluster_label = np.argmax(means)
            noisy_clients = np.where(gmm_pred == noisy_cluster_label)[0]
            noisy_clients = set(list(noisy_clients))
            vote.append(noisy_clients)
        
        vote_counts = {}
        for v in vote:
            v_tuple = tuple(sorted(v))
            vote_counts[v_tuple] = vote_counts.get(v_tuple, 0) + 1
        most_common_vote = max(vote_counts, key=vote_counts.get)
        detected_noisy_clients = list(most_common_vote)
        detected_clean_clients = [i for i in range(len(client_models)) if i not in detected_noisy_clients]
        
        print(f"检测到的噪声客户端: {detected_noisy_clients}")
        print(f"检测到的干净客户端: {detected_clean_clients}")
        
        self.noisy_clients = detected_noisy_clients
        self.clean_clients = detected_clean_clients
        
        return detected_noisy_clients, detected_clean_clients

    def federated_train(self, rounds=50, local_epochs=5, learning_rate=0.001, stage1_rounds=None):
        print(f"开始联邦学习训练，共 {rounds} 轮")


        global_train_losses = []
        global_val_dices = []
        global_val_ious = []
        all_round_avg_losses = []

        for round_num in range(rounds):
            print(f"\n=== 联邦训练第 {round_num + 1} 轮 ===")

            selected_clients = self.clients
            print(f"依次训练 {len(selected_clients)} 个客户端")

            client_data_sizes = []
            client_models = []
            round_avg_losses = []
            for client in selected_clients:
                client.model.load_state_dict(self.global_model.state_dict())
                client_data_sizes.append(len(client.train_loader.dataset))
                print(f"客户端 {client.client_id} 数据量: {len(client.train_loader.dataset)}")
            for client in selected_clients:
                print(f"客户端 {client.client_id} 开始本地训练...")
                avg_loss = client.train_local(epochs=local_epochs, learning_rate=learning_rate)
                round_avg_losses.append(avg_loss)
                print(f"客户端 {client.client_id} 本地训练完成，平均Loss: {avg_loss:.4f}")
                client_models.append(copy.deepcopy(client.model))
                dice, iou = client.evaluate_local()
                print(f"客户端 {client.client_id} 本地评估 - Dice: {dice:.4f}, IoU: {iou:.4f}")

            if stage1_rounds is not None and round_num == stage1_rounds - 1:
                client_loaders = []
                for client in self.clients:
                    client_loader = {
                        'train': client.train_loader,
                        'val': client.val_loader,
                        'has_noise': client.has_noise
                    }
                    client_loaders.append(client_loader)
                detected_noisy, detected_clean = self.detect_noisy_clients_by_loss(client_models, client_loaders)
                actual_noisy = [client.client_id for client in self.clients if client.has_noise]
                actual_clean = [client.client_id for client in self.clients if not client.has_noise]
                
                print(f"实际噪声客户端: {actual_noisy}")
                print(f"实际干净客户端: {actual_clean}")
                
                if actual_noisy:
                    true_positives = len(set(detected_noisy) & set(actual_noisy)) 
                    false_positives = len(set(detected_noisy) & set(actual_clean))
                    total_correct = true_positives + len(set(detected_clean) & set(actual_clean))
                    total_clients = len(self.clients)
                    overall_accuracy = total_correct / total_clients
                    print(f"噪声客户端检测准确率: {overall_accuracy:.2f}")
                print(f"在第 {round_num + 1} 轮（第一阶段结束）后完成噪声客户端检测")
            overall_avg_loss = sum(round_avg_losses) / len(round_avg_losses) if round_avg_losses else 0
            all_round_avg_losses.append(overall_avg_loss)
            print(f"第 {round_num + 1} 轮整体平均Loss: {overall_avg_loss:.4f}")
            print("执行 FedAvg 聚合...")
            self.global_model = self.aggregate_weights_fedavg(client_models, client_data_sizes)
            if len(selected_clients) > 0:
                temp_test_loader = selected_clients[0].val_loader
                global_dice, global_iou = evaluate_model(self.global_model, temp_test_loader, self.device)

                global_val_dices.append(global_dice)
                global_val_ious.append(global_iou)

                print(f"全局模型评估 - Dice: {global_dice:.4f}, IoU: {global_iou:.4f}")

            print(f"第 {round_num + 1} 轮训练完成")

        return self.global_model, (global_val_dices, global_val_ious, all_round_avg_losses)