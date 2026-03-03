import os
import numpy as np
from PIL import Image

def preprocess_label(label, threshold=128):
    """
    将标签图像中的像素值映射为二分类 [0, 1]
    :param label: 输入标签图像 (numpy array)
    :param foreground_threshold: 判断前景的阈值
    :return: 处理后的标签图像
    """
    # 将大于阈值的像素设为前景 (1)，其余设为背景 (0)
    processed_label = np.where(label > threshold, 255, 0)
    return processed_label.astype(np.uint8)
def process_labels(input_folder, output_folder, method="threshold"):
    """
    批量处理标签图像
    :param input_folder: 输入标签文件夹路径
    :param output_folder: 输出标签文件夹路径
    :param method: 处理方法 ("threshold" 或 "mapping")
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 打开图像并转换为 NumPy 数组
            label = np.array(Image.open(input_path))
            
            if method == "threshold":
                processed_label = preprocess_label(label)

            # 保存处理后的标签图像
            Image.fromarray(processed_label).save(output_path)  # 指定模式为 'L'（灰度图）
            print(f"Processed {filename}")

# 示例用法
input_folder = "D:\\kvasir-seg\\train\\masks"
output_folder = "D:\\kvasir-seg\\train\\processed_labels"
process_labels(input_folder, output_folder, method="threshold")