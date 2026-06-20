import torch
import torch.nn as nn
from .all_models import get_model


class DualNetwork(nn.Module):
    """
    双网络结构，用于处理噪声客户端
    """
    def __init__(self, base_model1, base_model2):
        super(DualNetwork, self).__init__()
        self.network1 = base_model1
        self.network2 = base_model2
        # 设置与基础模型相同的属性
        if hasattr(base_model1, 'n_channels'):
            self.n_channels = base_model1.n_channels
        if hasattr(base_model1, 'n_classes'):
            self.n_classes = base_model1.n_classes

    def forward(self, x):
        output1 = self.network1(x)
        output2 = self.network2(x)
        # 可以采用多种融合策略，这里是简单的平均
        combined_output = (output1 + output2) / 2
        return combined_output


class DualUNet(DualNetwork):
    """
    专门用于UNet的双网络实现
    """
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        from .unet import UNet
        model1 = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
        model2 = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
        super(DualUNet, self).__init__(model1, model2)


class DualSwinUNet(DualNetwork):
    """
    专门用于SwinUNet的双网络实现
    """
    def __init__(self, img_size=256, num_classes=3):
        from .swin_unet import SwinUnet
        model1 = SwinUnet(img_size=img_size, num_classes=num_classes)
        model2 = SwinUnet(img_size=img_size, num_classes=num_classes)
        super(DualSwinUNet, self).__init__(model1, model2)


def get_dual_model(model_name, **kwargs):
    """
    获取双网络模型
    """
    if model_name == 'DualUNet':
        return DualUNet(
            n_channels=kwargs.get('n_channels', 3),
            n_classes=kwargs.get('n_classes', 3),
            bilinear=kwargs.get('bilinear', True)
        )
    elif model_name == 'DualSwinUNet':
        return DualSwinUNet(
            img_size=kwargs.get('img_size', 256),
            num_classes=kwargs.get('num_classes', 3)
        )
    else:
        # 通用双网络
        base_model = get_model(model_name, kwargs.get('pretrained', False))
        base_model_copy = get_model(model_name, kwargs.get('pretrained', False))
        return DualNetwork(base_model, base_model_copy)