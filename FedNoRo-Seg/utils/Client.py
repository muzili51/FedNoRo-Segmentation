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
        # 创建双网络模型
        dual_model = DualNetwork(model1, model2)
        super().__init__(client_id, dual_model, train_loader, val_loader, device, has_noise)
        self.model1 = model1
        self.model2 = model2

    def train_local(self, epochs=5, learning_rate=0.001):
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.model1.train()
        self.model2.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                   lr=learning_rate)

        running_loss = 0.0
        for epoch in range(epochs):
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                output1 = self.model1(images)
                output2 = self.model2(images)
                loss1 = criterion(output1, masks)
                loss2 = criterion(output2, masks)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss