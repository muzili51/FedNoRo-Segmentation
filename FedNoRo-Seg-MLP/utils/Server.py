import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from .Client import evaluate_model
import numpy as np
from sklearn.mixture import GaussianMixture
from .Client import evaluate_model, DualNetworkClient


class FedAvgServer:
    def __init__(self, global_model, clients, device):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.noisy_clients = []
        self.clean_clients = []
        self.feature_buffer = {}  # 存储客户端特征
    
    def aggregate_weights_fedavg(self, client_models, client_data_sizes):
        """标准 FedAvg 参数聚合"""
        total_data_size = sum(client_data_sizes)
        aggregated_state_dict = OrderedDict()
        
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
    
    def extract_bottleneck_features(self, model, dataloader, device, num_samples=100):
        """
        提取模型的 bottleneck 特征表示
        Returns:
            features: [N, C, H, W] 特征集合
        """
        model.eval()
        features_list = []
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break
                
                images = images.to(device)
                
                if hasattr(model, 'get_bottleneck_features'):
                    feat = model.get_bottleneck_features(images)
                else:
                    # 对于双网络，获取拼接特征
                    feat = model.get_bottleneck_features(images)
                
                features_list.append(feat.cpu())
        
        if features_list:
            features = torch.cat(features_list, dim=0)
            return features
        return None
    
    def aggregate_features(self, client_models, client_data_sizes, validation_loader):
        """
        聚合客户端的 bottleneck 特征表示
        用于增强全局模型的特征表示能力
        """
        print("开始特征聚合...")
        
        all_features = []
        all_weights = []
        
        for client_idx, client_model in enumerate(client_models):
            # 提取特征
            features = self.extract_bottleneck_features(
                client_model, validation_loader, self.device
            )
            
            if features is not None:
                all_features.append(features)
                all_weights.append(client_data_sizes[client_idx])
        
        if not all_features:
            print("无特征可聚合")
            return None
        
        # 加权平均特征
        total_weight = sum(all_weights)
        normalized_weights = [w / total_weight for w in all_weights]
        
        # 对齐特征维度并聚合
        aggregated_features = None
        for feat, weight in zip(all_features, normalized_weights):
            if aggregated_features is None:
                aggregated_features = weight * feat
            else:
                # 需要确保特征维度一致
                if aggregated_features.shape == feat.shape:
                    aggregated_features += weight * feat
        
        print(f"聚合特征形状: {aggregated_features.shape if aggregated_features is not None else None}")
        
        return aggregated_features
    
    def aggregate_weights_daagg(self, client_models, client_data_sizes, client_losses, 
                                client_features=None):
        """
        动态聚合 + 特征聚合
        Args:
            client_features: 客户端 bottleneck 特征列表
        """
        print("执行 DaAgg 动态聚合...")
        
        # 1. 基于损失的权重计算
        client_losses = np.array(client_losses)
        min_loss = np.min(client_losses)
        max_loss = np.max(client_losses)
        
        if max_loss == min_loss:
            client_weights = np.ones(len(client_losses)) / len(client_losses)
        else:
            normalized_losses = (client_losses - min_loss) / (max_loss - min_loss + 1e-8)
            performance_scores = 1 / (normalized_losses + 1e-8)
            client_weights = performance_scores / np.sum(performance_scores)
        
        # 2. 对噪声客户端降权
        for client_idx in self.noisy_clients:
            client_weights[client_idx] *= 0.5
        
        # 重新归一化
        client_weights = client_weights / np.sum(client_weights)
        
        print(f"客户端损失: {client_losses}")
        print(f"客户端权重: {client_weights}")
        
        # 3. 参数聚合
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
        
        # 4. 特征聚合（可选，用于特征对齐或蒸馏）
        if client_features is not None:
            aggregated_feat = self.aggregate_features(
                client_models, client_data_sizes, self.clients[0].val_loader
            )
            if aggregated_feat is not None:
                self.feature_buffer['global'] = aggregated_feat
                print("特征聚合完成")
        
        return self.global_model
    
    def detect_noisy_clients_by_loss(self, client_models, client_loaders):
        """基于损失和特征一致性的噪声客户端检测"""
        print("\n=== 检测噪声客户端 ===")
        
        self.global_model.eval()
        n_classes = self.global_model.n_classes
        
        # 1. 收集每客户端的损失指标
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
        
        # 2. 收集特征一致性指标
        feature_consistency = np.zeros(len(client_models))
        
        for client_id, client_model in enumerate(client_models):
            features = self.extract_bottleneck_features(
                client_model, client_loaders[client_id]['val'], self.device
            )
            
            if features is not None and 'global' in self.feature_buffer:
                global_feat = self.feature_buffer['global']
                # 计算与全局特征的相似度
                feat_flat = features.view(features.shape[0], -1)
                global_flat = global_feat.view(global_feat.shape[0], -1)
                
                # 余弦相似度
                similarity = F.cosine_similarity(
                    feat_flat.mean(dim=0, keepdim=True),
                    global_flat.mean(dim=0, keepdim=True),
                    dim=1
                ).item()
                
                feature_consistency[client_id] = similarity
        
        # 3. 融合损失和特征指标
        if feature_consistency.max() > 0:
            normalized_consistency = feature_consistency / (feature_consistency.max() + 1e-8)
            # 特征一致性低的客户端更可能是噪声客户端
            metrics = metrics * (1 - normalized_consistency).reshape(-1, 1)
        
        # 4. GMM 聚类检测
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
            vote.append(set(list(noisy_clients)))
        
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
        """联邦训练主流程"""
        print(f"开始联邦学习训练，共 {rounds} 轮")
        
        global_train_losses = []
        global_val_dices = []
        global_val_ious = []
        all_round_avg_losses = []
        
        for round_num in range(rounds):
            print(f"\n=== 联邦训练第 {round_num + 1} 轮 ===")
            
            selected_clients = self.clients
            client_data_sizes = []
            client_models = []
            client_features = []
            round_avg_losses = []
            
            # 分发全局模型 - 修复：根据客户端类型分别处理
            for client in selected_clients:
                if isinstance(client, DualNetworkClient):
                    client.model1.load_state_dict(self.global_model.state_dict())
                else:
                    client.model.load_state_dict(self.global_model.state_dict())
                client_data_sizes.append(len(client.train_loader.dataset))
                
                client_data_sizes.append(len(client.train_loader.dataset))
            
            # 客户端训练
            for client in selected_clients:
                print(f"客户端 {client.client_id} 开始本地训练...")
                
                if isinstance(client, DualNetworkClient):
                    avg_loss = client.train_local(epochs=local_epochs, learning_rate=learning_rate)
                    # 双网络客户端返回 model1 作为聚合模型
                    client_models.append(copy.deepcopy(client.model1))
                else:
                    avg_loss = client.train_local(epochs=local_epochs, learning_rate=learning_rate)
                    client_models.append(copy.deepcopy(client.model))
                
                round_avg_losses.append(avg_loss)
                
                # 提取特征
                if isinstance(client, DualNetworkClient):
                    features = self.extract_bottleneck_features(
                        client.model1, client.val_loader, self.device
                    )
                else:
                    features = self.extract_bottleneck_features(
                        client.model, client.val_loader, self.device
                    )
                client_features.append(features)
                
                dice, iou = client.evaluate_local()
                print(f"客户端 {client.client_id} - Loss: {avg_loss:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}")
            
            # 第一阶段结束检测噪声客户端
            if stage1_rounds is not None and round_num == stage1_rounds - 1:
                client_loaders = []
                for client in self.clients:
                    client_loaders.append({
                        'train': client.train_loader,
                        'val': client.val_loader,
                        'has_noise': client.has_noise
                    })
                
                detected_noisy, detected_clean = self.detect_noisy_clients_by_loss(
                    client_models, client_loaders
                )
                
                actual_noisy = [c.client_id for c in self.clients if c.has_noise]
                actual_clean = [c.client_id for c in self.clients if not c.has_noise]
                
                if actual_noisy:
                    true_positives = len(set(detected_noisy) & set(actual_noisy))
                    overall_accuracy = (true_positives + len(set(detected_clean) & set(actual_clean))) / len(self.clients)
                    print(f"噪声客户端检测准确率：{overall_accuracy:.2f}")
            
            # 聚合
            overall_avg_loss = sum(round_avg_losses) / len(round_avg_losses) if round_avg_losses else 0
            all_round_avg_losses.append(overall_avg_loss)
            
            print("执行 DaAgg 动态聚合 + 特征聚合...")
            self.global_model = self.aggregate_weights_daagg(
                client_models, client_data_sizes, round_avg_losses, client_features
            )
            
            # 全局评估
            if len(selected_clients) > 0:
                temp_test_loader = selected_clients[0].val_loader
                global_dice, global_iou = evaluate_model(self.global_model, temp_test_loader, self.device)
                global_val_dices.append(global_dice)
                global_val_ious.append(global_iou)
                print(f"全局模型评估 - Dice: {global_dice:.4f}, IoU: {global_iou:.4f}")
            
            print(f"第 {round_num + 1} 轮训练完成")
        
        return self.global_model, (global_val_dices, global_val_ious, all_round_avg_losses)