# utils/Server.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from .Client import evaluate_model
import numpy as np
from sklearn.mixture import GaussianMixture
from utils.options_for_消融试验1 import args_parser
args = args_parser()
class FedAvgServer:
    def __init__(self, global_model, clients, device):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.overall_recall = []
        self.overall_precision = []
        self.overall_match_ratio=[]
        self.noisy_clients = []  # 存储检测到的噪声客户端
        self.clean_clients = []  # 存储检测到的干净客户端
        self.overall_noise_clients_detect_performance  = [] # 存储总体噪声客户端的检测性能

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
    
    def evaluate_client_performance(self, client_models, validation_loader):
        """
        评估每个客户端模型在验证集上的性能
        返回每个客户端的损失值
        """
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
                    
                    # 确保标签值在有效范围内
                    if hasattr(client_model, 'n_classes'):
                        if masks.min() < 0 or masks.max() >= client_model.n_classes:
                            masks = torch.clamp(masks, 0, client_model.n_classes - 1)
                    
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            performances.append(avg_loss)
            
            print(f"客户端 {i} 验证损失: {avg_loss:.4f}")
        
        # 恢复原始模型状态
        self.global_model = original_model
        
        return performances


    def aggregate_weights_daagg(self, client_models, client_data_sizes, client_losses):
        
        # 计算每个客户端的权重，性能越好权重越高
        # 将损失值转换为权重（损失越小权重越大）
        client_losses = np.array(client_losses)
        
        # 避免除零错误，计算相对性能权重
        min_loss = np.min(client_losses)
        max_loss = np.max(client_losses)
        
        if max_loss == min_loss:
            # 所有客户端性能相同，使用均匀权重
            client_weights = np.ones(len(client_losses)) / len(client_losses)
        else:
            # 使用归一化后的倒数来表示性能权重
            normalized_losses = (client_losses - min_loss) / (max_loss - min_loss)
            performance_scores = 1 / (normalized_losses + 1e-8)  # 加小常数避免除零
            client_weights = performance_scores / np.sum(performance_scores)
        
        print(f"客户端损失: {client_losses}")
        print(f"客户端权重: {client_weights}")
        
        # 计算加权聚合
        total_data_size = sum(client_data_sizes)
        aggregated_state_dict = OrderedDict()
        
        # 获取全局模型参数键值
        global_keys = self.global_model.state_dict().keys()
        
        for param_idx, param_name in enumerate(global_keys):
            weighted_param = None
            
            for client_idx, client_model in enumerate(client_models):
                client_param = list(client_model.state_dict().values())[param_idx]
                
                # 结合数据量权重和性能权重
                data_weight = client_data_sizes[client_idx] / total_data_size
                combined_weight = client_weights[client_idx]  # 只使用性能权重，也可以组合
                
                if weighted_param is None:
                    weighted_param = combined_weight * client_param.clone()
                else:
                    weighted_param += combined_weight * client_param.clone()
            
            aggregated_state_dict[param_name] = weighted_param
        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated_state_dict)
        
        return self.global_model
    
    def evaluate_client_performance(self, client_models, validation_loader):
        """
        评估每个客户端模型在验证集上的性能
        返回每个客户端的损失值
        """
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
                    
                    # 确保标签值在有效范围内
                    if hasattr(client_model, 'n_classes'):
                        if masks.min() < 0 or masks.max() >= client_model.n_classes:
                            masks = torch.clamp(masks, 0, client_model.n_classes - 1)
                    
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            performances.append(avg_loss)
            
            print(f"客户端 {i} 验证损失: {avg_loss:.4f}")
        
        # 恢复原始模型状态
        self.global_model = original_model
        
        return performances

    def aggregate_weights_daagg(self, client_models, client_data_sizes, client_losses):
        """
        DaAgg (Dynamic Aggregation) 权重聚合算法
        基于每个客户端模型性能的动态加权聚合
        """
        print("执行 DaAgg 动态聚合...")
        
        # 计算每个客户端的权重，性能越好权重越高
        # 将损失值转换为权重（损失越小权重越大）
        client_losses = np.array(client_losses)
        
        # 避免除零错误，计算相对性能权重
        min_loss = np.min(client_losses)
        max_loss = np.max(client_losses)
        
        if max_loss == min_loss:
            # 所有客户端性能相同，使用均匀权重
            client_weights = np.ones(len(client_losses)) / len(client_losses)
        else:
            # 使用归一化后的倒数来表示性能权重
            normalized_losses = (client_losses - min_loss) / (max_loss - min_loss + 1e-8)  # 加小常数避免除零
            performance_scores = 1 / (normalized_losses + 1e-8)  # 加小常数避免除零
            client_weights = performance_scores / np.sum(performance_scores)
        
        print(f"客户端损失: {client_losses}")
        print(f"客户端权重: {client_weights}")
        
        # 计算加权聚合
        total_data_size = sum(client_data_sizes)
        aggregated_state_dict = OrderedDict()
        
        # 获取全局模型参数键值
        global_keys = self.global_model.state_dict().keys()
        
        for param_idx, param_name in enumerate(global_keys):
            weighted_param = None
            
            for client_idx, client_model in enumerate(client_models):
                client_param = list(client_model.state_dict().values())[param_idx]
                
                # 使用性能权重
                combined_weight = client_weights[client_idx]
                
                if weighted_param is None:
                    weighted_param = combined_weight * client_param.clone()
                else:
                    weighted_param += combined_weight * client_param.clone()
            
            aggregated_state_dict[param_name] = weighted_param
        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated_state_dict)
        
        return self.global_model
    def evaluate_client_performance(self, client_models, validation_loader):
        """
        评估每个客户端模型在验证集上的性能
        返回每个客户端的损失值
        """
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
                    
                    # 确保标签值在有效范围内
                    if masks.min() < 0 or masks.max() >= client_model.n_classes:
                        masks = torch.clamp(masks, 0, client_model.n_classes - 1)
                    
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            performances.append(avg_loss)
            
            print(f"客户端 {i} 验证损失: {avg_loss:.4f}")
        
        # 恢复原始模型状态
        self.global_model = original_model
        
        return performances

    def detect_noisy_clients_by_loss(self, client_models, client_loaders):
        """
        通过损失函数来检测噪声客户端
        """
        print("\n=== 检测噪声客户端 ===")
        
        # 设置模型为评估模式
        self.global_model.eval()
        
        # 为每个客户端计算损失统计
        n_classes = self.global_model.n_classes
        metrics = np.zeros((len(client_models), n_classes))
        num = np.zeros((len(client_models), n_classes))
        
        for client_id, (client_model, client_loader) in enumerate(zip(client_models, client_loaders)):
            client_model.eval()
            client_model.to(self.device)
            
            # 使用reduction='none'来获得每个样本的损失
            criterion = nn.CrossEntropyLoss(reduction='none')
            
            losses_per_class = [[] for _ in range(n_classes)]
            
            with torch.no_grad():
                for images, masks in client_loader['train']:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = client_model(images)
                    
                    # 确保标签值在有效范围内
                    if masks.min() < 0 or masks.max() >= n_classes:
                        masks = torch.clamp(masks, 0, n_classes - 1)
                    
                    # 计算每个样本的损失
                    per_sample_loss = criterion(outputs, masks)
                    
                    # 按类别收集损失
                    for cls in range(n_classes):
                        class_mask = (masks == cls)
                        class_losses = per_sample_loss[class_mask]
                        if class_losses.numel() > 0:
                            losses_per_class[cls].extend(class_losses.cpu().numpy())
            
            # 计算每个客户端每个类别的平均损失
            for cls in range(n_classes):
                if len(losses_per_class[cls]) > 0:
                    metrics[client_id, cls] = np.mean(losses_per_class[cls])
                    num[client_id, cls] = len(losses_per_class[cls])
                else:
                    metrics[client_id, cls] = 0
        
        # 避免除零错误，计算每个类别的平均损失
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                if num[i, j] > 0:
                    metrics[i, j] = metrics[i, j] / num[i, j]
                else:
                    # 如果某个客户端某类没有样本，用其他客户端该类的最小值填充
                    if np.any(metrics[:, j] > 0):
                        metrics[i, j] = np.min(metrics[metrics[:, j] > 0])
                    else:
                        metrics[i, j] = 0
        
        # 处理NaN值
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                if np.isnan(metrics[i, j]):
                    if np.any(~np.isnan(metrics[:, j])):
                        metrics[i, j] = np.nanmin(metrics[:, j])
                    else:
                        metrics[i, j] = 0
        
        # 对每个类别进行归一化
        for j in range(metrics.shape[1]):
            min_val = metrics[:, j].min()
            max_val = metrics[:, j].max()
            if max_val != min_val:
                metrics[:, j] = (metrics[:, j] - min_val) / (max_val - min_val)
            else:
                # 如果所有值都相同，设为0
                metrics[:, j] = 0
        
        print("客户端指标矩阵:")
        print(metrics)
        
        # 使用GMM进行聚类
        vote = []
        for i in range(9):  # 进行多次投票
            gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
            gmm_pred = gmm.predict(metrics)
            
            # 确定哪个聚类是噪声客户端（通常损失更高的那个）
            means = gmm.means_.sum(axis=1)
            noisy_cluster_label = np.argmax(means)
            noisy_clients = np.where(gmm_pred == noisy_cluster_label)[0]
            noisy_clients = set(list(noisy_clients))
            vote.append(noisy_clients)
        
        # 统计投票结果
        vote_counts = {}
        for v in vote:
            v_tuple = tuple(sorted(v))
            vote_counts[v_tuple] = vote_counts.get(v_tuple, 0) + 1
        
        # 选择得票最多的方案
        most_common_vote = max(vote_counts, key=vote_counts.get)
        detected_noisy_clients = list(most_common_vote)
        detected_clean_clients = [i for i in range(len(client_models)) if i not in detected_noisy_clients]
        
        print(f"检测到的噪声客户端: {detected_noisy_clients}")
        print(f"检测到的干净客户端: {detected_clean_clients}")
        
        self.noisy_clients = detected_noisy_clients
        self.clean_clients = detected_clean_clients
        
        return detected_noisy_clients, detected_clean_clients

    def federated_train(self, rounds=50, local_epochs=5, learning_rate=0.001, stage1_rounds=None):
        """
        联邦学习训练主循环 - 每个客户端依次训练
        """
        print(f"开始联邦学习训练，共 {rounds} 轮")

        # 存储训练历史
        global_train_losses = []
        global_val_dices = []
        global_val_ious = []
        all_round_avg_losses = []

        for round_num in range(rounds):
            print(f"\n=== 联邦训练第 {round_num + 1} 轮 ===")

            # 依次训练所有客户端
            selected_clients = self.clients  # 训练所有客户端，不再随机选择
            print(f"依次训练 {len(selected_clients)} 个客户端")

            # 收集客户端数据大小
            client_data_sizes = []
            client_models = []
            round_avg_losses = []  # 新增：记录当前轮次每个客户端的平均loss

            # 将全局模型分发给所有客户端
            for client in selected_clients:
                # 复制全局模型到客户端
                client.model.load_state_dict(self.global_model.state_dict())

                # 记录客户端数据大小
                client_data_sizes.append(len(client.train_loader.dataset))
                print(f"客户端 {client.client_id} 数据量: {len(client.train_loader.dataset)}")

            # 客户端依次本地训练
            for client in selected_clients:
                print(f"客户端 {client.client_id} 开始本地训练...")
                avg_loss = client.train_local(epochs=local_epochs, learning_rate=learning_rate)
                
                # 新增：记录并输出当前客户端的平均loss
                round_avg_losses.append(avg_loss)
                print(f"客户端 {client.client_id} 本地训练完成，平均Loss: {avg_loss:.4f}")

                # 收集训练后的模型
                client_models.append(copy.deepcopy(client.model))

                # 评估客户端模型
                dice, iou = client.evaluate_local()
                print(f"客户端 {client.client_id} 本地评估 - Dice: {dice:.4f}, IoU: {iou:.4f}")

            if stage1_rounds is not None:
                # 获取客户端加载器列表用于检测
                client_loaders = []
                for client in self.clients:
                    # 创建一个包含训练和验证加载器的字典
                    client_loader = {
                        'train': client.train_loader,
                        'val': client.val_loader,
                        'has_noise': client.has_noise
                    }
                    client_loaders.append(client_loader)
                
                # 检测噪声客户端
                detected_noisy, detected_clean = self.detect_noisy_clients_by_loss(client_models, client_loaders)
                
                # 输出比较
                actual_noisy = [client.client_id for client in self.clients if client.has_noise]
                actual_clean = [client.client_id for client in self.clients if not client.has_noise]
                
                print(f"实际噪声客户端: {actual_noisy}")
                print(f"实际干净客户端: {actual_clean}")
                
                if actual_noisy:
                    true_positives = len(set(detected_noisy) & set(actual_noisy))  # 正确识别的噪声客户端
                    false_positives = len(set(detected_noisy) & set(actual_clean))  # 错误识别的干净客户端为噪声
                    false_negatives = len(set(detected_clean) & set(actual_noisy))  # 未识别出的噪声客户端
                    true_negatives = len(set(detected_clean) & set(actual_clean))   # 正确识别的干净客户端

                    # 计算 Recall, Precision, Match Ratio
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    match_ratio = (true_positives + true_negatives) / args.n_clients
                    self.overall_recall.append(recall)
                    self.overall_precision.append(precision)
                    self.overall_match_ratio.append(match_ratio)
                    true_positives = len(set(detected_noisy) & set(actual_noisy))  # 正确识别的噪声客户端
                    false_positives = len(set(detected_noisy) & set(actual_clean))  # 错误识别的干净客户端为噪声
                    # 计算总体准确率
                    total_correct = true_positives + len(set(detected_clean) & set(actual_clean))
                    total_clients = len(self.clients)
                    overall_accuracy = total_correct / total_clients
                    print(f"噪声客户端检测准确率: {overall_accuracy:.2f}")
                    self.overall_noise_clients_detect_performance.append(overall_accuracy)

            # 计算并输出当前轮次的平均loss
            overall_avg_loss = sum(round_avg_losses) / len(round_avg_losses) if round_avg_losses else 0
            all_round_avg_losses.append(overall_avg_loss)
            print(f"第 {round_num + 1} 轮整体平均Loss: {overall_avg_loss:.4f}")

            # 使用 FedAvg 聚合客户端模型
            print("执行 FedAvg 聚合...")
            self.global_model = self.aggregate_weights_fedavg(client_models, client_data_sizes)

            # 评估全局模型
            # 创建临时测试加载器（这里使用第一个客户端的验证集作为全局评估）
            if len(selected_clients) > 0:
                temp_test_loader = selected_clients[0].val_loader
                global_dice, global_iou = evaluate_model(self.global_model, temp_test_loader, self.device)

                global_val_dices.append(global_dice)
                global_val_ious.append(global_iou)

                print(f"全局模型评估 - Dice: {global_dice:.4f}, IoU: {global_iou:.4f}")

            print(f"第 {round_num + 1} 轮训练完成")

        return self.global_model, (global_val_dices, global_val_ious, all_round_avg_losses)