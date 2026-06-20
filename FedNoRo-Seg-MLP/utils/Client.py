import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dual_network import DualNetwork


class Client:
    def __init__(self, client_id, model, train_loader, val_loader, device, has_noise=False):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.has_noise = has_noise

    def train_local(self, epochs=5, learning_rate=0.001):
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

                if masks.min() < 0 or masks.max() >= self.model.n_classes:
                    masks = torch.clamp(masks, 0, self.model.n_classes - 1)

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def evaluate_local(self):
        return evaluate_model(self.model, self.val_loader, self.device)


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

def calculate_metrics(pred, target, num_classes=3):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum().item()
    total = target.numel()
    accuracy = correct / total
    iou_list = []
    dice_list = []

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
        pred_area = pred_i.sum().item()
        target_area = target_i.sum().item()
        
        if pred_area + target_area == 0:
            dice = 1.0
        else:
            dice = (2 * intersection) / (pred_area + target_area)
        dice_list.append(dice)

    mean_iou = sum(iou_list) / len(iou_list)
    mean_dice = sum(dice_list) / len(dice_list)
    
    return accuracy, mean_iou, mean_dice


def evaluate_model(model, test_loader, device):
    model.eval()
    model = model.to(device)
    total_accuracy = 0
    total_iou = 0
    total_dice = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            accuracy = calculate_accuracy(outputs, masks)
            total_accuracy += accuracy

            _, mean_iou, mean_dice = calculate_metrics(outputs, masks)
            total_iou += mean_iou
            total_dice += mean_dice

            num_batches += 1

    avg_accuracy = total_accuracy / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches

    return avg_dice, avg_iou

class DualNetworkClient(Client):
    def __init__(self, client_id, model1, model2, train_loader, val_loader, device, has_noise=False):
        dual_model = DualNetwork(model1, model2)
        super().__init__(client_id, dual_model, train_loader, val_loader, device, has_noise)
        self.model1 = model1
        self.model2 = model2
        self.prev_prediction_unet = None
        self.prev_prediction_transformer = None
    
    def calculate_temporal_consistency_dual(self, current_pred_unet, current_pred_transformer):
        """双网络时序一致性检查"""
        consistency_scores = []
        
        # U-Net 时序一致性
        if self.prev_prediction_unet is not None:
            diff_unet = torch.abs(current_pred_unet.float() - self.prev_prediction_unet.float())
            consistency_scores.append(1.0 - diff_unet.sum() / diff_unet.numel())
        
        # Transformer 时序一致性
        if self.prev_prediction_transformer is not None:
            diff_trans = torch.abs(current_pred_transformer.float() - self.prev_prediction_transformer.float())
            consistency_scores.append(1.0 - diff_trans.sum() / diff_trans.numel())
        
        # 双网络间一致性
        diff_dual = torch.abs(current_pred_unet.float() - current_pred_transformer.float())
        consistency_scores.append(1.0 - diff_dual.sum() / diff_dual.numel())
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
    
    def train_local(self, epochs=5, learning_rate=0.001, gamma=0.5):
        device = self.device
        model1 = self.model1
        model2 = self.model2
        train_loader = self.train_loader
        
        model1.to(device)
        model2.to(device)
        model1.train()
        model2.train()
        
        criterion_sup = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
            list(model1.parameters()) + list(model2.parameters()),
            lr=learning_rate
        )
        
        running_loss = 0.0
        total_batches = 0
        
        for epoch in range(epochs):
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                B, H, W = masks.shape
                
                optimizer.zero_grad()
                
                # 双网络前向传播
                logits1 = model1(images)
                logits2 = model2(images)
                
                # 监督损失
                sup_loss1 = criterion_sup(logits1, masks)
                sup_loss2 = criterion_sup(logits2, masks)
                
                # 时序一致性噪声检测
                with torch.no_grad():
                    pred1 = torch.argmax(logits1, dim=1)
                    pred2 = torch.argmax(logits2, dim=1)
                    
                    consistency_score = self.calculate_temporal_consistency_dual(pred1, pred2)
                    
                    # 更新上一帧预测
                    self.prev_prediction_unet = pred1.clone()
                    self.prev_prediction_transformer = pred2.clone()
                
                # 基于一致性分数调整损失权重
                noise_weight = 1.0 if consistency_score > 0.7 else 0.5
                
                # 互学习损失
                prob1 = F.softmax(logits1, dim=1)
                prob2 = F.softmax(logits2, dim=1)
                
                log_pred1 = F.log_softmax(logits1, dim=1)
                log_pred2 = F.log_softmax(logits2, dim=1)
                
                kl1 = F.kl_div(log_pred1, prob2, reduction='none').sum(dim=1, keepdim=True)
                kl2 = F.kl_div(log_pred2, prob1, reduction='none').sum(dim=1, keepdim=True)
                
                unsup_loss1 = (kl1 * noise_weight).mean()
                unsup_loss2 = (kl2 * noise_weight).mean()
                unsup_loss = 0.5 * (unsup_loss1 + unsup_loss2)
                
                total_sup_loss = 0.5 * (sup_loss1.mean() + sup_loss2.mean())
                total_loss = total_sup_loss + gamma * unsup_loss
                
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item()
                total_batches += 1
        
        avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
        return avg_loss