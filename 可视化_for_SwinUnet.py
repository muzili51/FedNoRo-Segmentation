import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Unet_Test import UNet  # 假设这是你的 UNet 模型定义
from model.swin_unet import SwinUnet
# 1. 定义模型结构
model = UNet(n_channels=3, n_classes=3)
model.eval()  # 设置为评估模式

# 2. 加载模型权重（注意：这里是纯权重文件）
model_save_path = r"C:\\Users\\muzili\\Desktop\\saved_models_for_swin_unet_kvasir\\SwinUnet_model_noise_0.8.pth"
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

# 3. 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),       # 调整图像大小
    transforms.ToTensor(),               # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 4. 读取并预处理图像
image_path = "D:\\kvasir-seg\\train\\images\\cju2t62nq45jl0799odpufwx6.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # 添加批次维度

# 5. 推理预测
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # 获取分割结果

# 6. 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmentation Result")
plt.imshow(prediction, cmap="jet")
plt.axis("off")

plt.tight_layout()
plt.show()