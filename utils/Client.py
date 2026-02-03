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
    """
    计算准确率、IoU和Dice系数
    """
    pred = torch.argmax(pred, dim=1)
    
    # 计算准确率
    correct = (pred == target).sum().item()
    total = target.numel()
    accuracy = correct / total
    
    # 计算IoU和Dice
    iou_list = []
    dice_list = []

    for i in range(num_classes):
        pred_i = (pred == i)
        target_i = (target == i)

        intersection = (pred_i & target_i).sum().item()
        union = (pred_i | target_i).sum().item()

        # 计算IoU
        if union == 0:
            iou = 1.0  # 如果预测和目标都没有此类别，则IoU设为1
        else:
            iou = intersection / union
        iou_list.append(iou)

        # 计算Dice系数
        pred_area = pred_i.sum().item()
        target_area = target_i.sum().item()
        
        if pred_area + target_area == 0:
            dice = 1.0  # 如果预测和目标都没有此类别，则Dice设为1
        else:
            dice = (2 * intersection) / (pred_area + target_area)
        dice_list.append(dice)

    mean_iou = sum(iou_list) / len(iou_list)
    mean_dice = sum(dice_list) / len(dice_list)
    
    return accuracy, mean_iou, mean_dice


def evaluate_model(model, test_loader, device):
    """
    在测试集上评估模型，返回准确率、平均IoU和平均Dice系数
    """
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
    """
    使用双网络的客户端
    """
    def __init__(self, client_id, model1, model2, train_loader, val_loader, device, has_noise=False):
        # 创建双网络模型
        dual_model = DualNetwork(model1, model2)
        super().__init__(client_id, dual_model, train_loader, val_loader, device, has_noise)
        self.model1 = model1
        self.model2 = model2

    def train_local(self, epochs=5, learning_rate=0.001):
        """
        双网络本地训练
        """
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.model1.train()
        self.model2.train()

        criterion = nn.CrossEntropyLoss()
        # 优化两个子网络的参数
        optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                   lr=learning_rate)

        running_loss = 0.0
        for epoch in range(epochs):
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()

                # 分别获取两个网络的输出
                output1 = self.model1(images)
                output2 = self.model2(images)

                # 计算两个网络的损失
                loss1 = criterion(output1, masks)
                loss2 = criterion(output2, masks)

                # 总损失（可以调整权重）
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss