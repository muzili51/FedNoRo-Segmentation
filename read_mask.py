from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取灰度图片
image_path = r"D:\\Fundus-doFE\\Fundus\Domain1\\train\\mask\\g0027.png"  # 替换为你的图片路径
gray_image = Image.open(image_path).convert("L")  # 转换为灰度图像
gray_array = np.array(gray_image)  # 转换为 NumPy 数组

# 2. 创建 RGB 图像数组
height, width = gray_array.shape
rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

# 3. 映射灰度值到颜色
rgb_array[gray_array == 0] = [0, 0, 255]      # 0 -> 红色
rgb_array[gray_array == 128] = [0, 255, 0]    # 128 -> 绿色
rgb_array[gray_array == 255] = [255, 0, 0]    # 255 -> 蓝色

# 4. 转换为 PIL 图像
colored_image = Image.fromarray(rgb_array, mode="RGB")

# 5. 可视化结果
plt.figure(figsize=(10, 5))

# 显示原始灰度图像
plt.subplot(1, 2, 1)
plt.title("Original Grayscale Image")
plt.imshow(gray_image, cmap="gray")
plt.axis("off")

# 显示着色后的图像
plt.subplot(1, 2, 2)
plt.title("Colored Image")
plt.imshow(colored_image)
plt.axis("off")

plt.tight_layout()
plt.show()