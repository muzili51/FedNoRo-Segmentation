import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
from .unet import UNet, OutConv
from .swin_unet import SwinUnet

def get_model(model_name, pretrained=False):
    """Returns a CNN model
    Args:
      model_name: model name
      pretrained: True or False
    Returns:
      model: the desired model
    Raises:
      ValueError: If model name is not recognized.
    """
    if model_name == 'UNet':
        return UNet(n_channels=3, n_classes=3)
    elif model_name == 'SwinUNet':
        return SwinUnet(img_size=256, num_classes=3)
    else:
        raise ValueError('Name of model unknown %s' % model_name)


def modify_last_layer(model_name, model, num_classes, normed=False, bias=True):
    """modify the last layer of the CNN model to fit the num_classes
    Args:
      model_name: model name
      model: CNN model
      num_classes: class number
    Returns:
      model: the desired model
    """

    if 'UNet' in model_name:
        model.outc = OutConv(64, num_classes)
        last_layer = model.outc
    elif 'SwinUnet' in model_name or model_name == 'SwinUnet':
        model.outc = nn.Conv2d(model.swin_unet.num_features, num_classes, kernel_size=1)
        last_layer = model.outc
    else:
        raise NotImplementedError
    return model, last_layer


def classifier(num_features, num_classes):
    last_linear = nn.Linear(num_features, num_classes)
    return last_linear