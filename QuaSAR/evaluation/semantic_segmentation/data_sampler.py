# import os
# import random
# import shutil
#
#
# def clear_directory(directory):
#     """
#     Clear all files in a given directory.
#
#     Args:
#     - directory (str): The directory path to clear.
#     """
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         try:
#             if os.path.isdir(file_path):
#                 shutil.rmtree(file_path)  # Remove directory and its contents
#             else:
#                 os.remove(file_path)  # Remove file
#         except Exception as e:
#             print(f"Error while deleting {file_path}: {e}")
#
#
# def copy_samples(input_root, output_root, total_samples=10, ratio=(0.8, 0.1, 0.1)):
#     """
#     Copy images and labels to a new directory with a specified sampling ratio.
#
#     Args:
#     - input_root (str): Root directory containing 'images/training' and 'annotations/training'.
#     - output_root (str): Root directory to store the selected images and labels.
#     - total_samples (int): Total number of samples to select (default is 10).
#     - ratio (tuple): Sampling ratio for high_quality, medium_quality, low_quality images (default is (0.8, 0.1, 0.1)).
#     """
#     images_input_root = os.path.join(input_root, 'images', 'training')
#     labels_input_root = os.path.join(input_root, 'annotations', 'training')
#
#     images_output_root = os.path.join(output_root, 'images', 'training')
#     labels_output_root = os.path.join(output_root, 'annotations', 'training')
#
#     # Clear the output directories before copying new files
#     if os.path.exists(images_output_root):
#         clear_directory(images_output_root)
#     else:
#         os.makedirs(images_output_root)
#
#     if os.path.exists(labels_output_root):
#         clear_directory(labels_output_root)
#     else:
#         os.makedirs(labels_output_root)
#
#     quality_folders = ['high_quality', 'medium_quality', 'low_quality']
#
#     # Calculate the number of samples for each quality category
#     total_high = int(total_samples * ratio[0])
#     total_medium = int(total_samples * ratio[1])
#     total_low = total_samples - total_high - total_medium
#
#     all_selected_images = []
#
#     for quality, num_samples in zip(quality_folders, [total_high, total_medium, total_low]):
#         # Get all image and label file paths for this quality
#         image_files = [f for f in os.listdir(os.path.join(images_input_root, quality)) if f.endswith('.png')]
#         random.shuffle(image_files)
#
#         # 选取指定数量的样本
#         selected_images = image_files[:num_samples]
#         all_selected_images.extend([(quality, img) for img in selected_images])
#
#         # 遍历并复制选中的图像及其对应标签
#     for quality, image_name in all_selected_images:
#         # 图像路径
#         image_path = os.path.join(images_input_root, quality, image_name)
#         # 标签路径：名称添加 `_1stHO`
#         label_name = image_name.replace('.png', '_1stHO.png')
#         label_path = os.path.join(labels_input_root, quality, label_name)
#
#         # 确保标签存在
#         if os.path.exists(label_path):
#             # 复制图像和标签到输出目录
#             shutil.copy(image_path, os.path.join(images_output_root, image_name))
#             shutil.copy(label_path, os.path.join(labels_output_root, label_name))
#         else:
#             print(f"Warning: Label not found for {image_name}. Skipping.")
#     print(f"Successfully copied {len(all_selected_images)} image and label pairs!")
#
# def generate_ratios(start=0.0, end=1.0, step=0.1):
#     """自动生成所有满足 high + medium + low == 1 的比例组合"""
#     ratios = []
#     steps = int((end - start) / step) + 1
#     for i in range(steps):
#         for j in range(steps):
#             high = round(start + i * step, 2)
#             medium = round(start + j * step, 2)
#             low = round(1.0 - high - medium, 2)
#             if 0.0 <= low <= 1.0:
#                 ratios.append((high, medium, low))
#     return ratios

import os
import random
import shutil


def generate_ratios(start=0.0, end=1.0, step=0.1):
    """自动生成所有满足 high + medium + low == 1 的比例组合。"""
    ratios = []
    steps = int((end - start) / step) + 1
    for i in range(steps):
        for j in range(steps):
            high = round(start + i * step, 2)
            medium = round(start + j * step, 2)
            low = round(1.0 - high - medium, 2)
            if 0.0 <= low <= 1.0:
                ratios.append((high, medium, low))
    return ratios


def clear_directory(directory):
    # ... (保持你原来的代码不变) ...
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        except Exception as e:
            print(f"Error while deleting {file_path}: {e}")


# 将这里的默认 ratio 修改为你需要的 (0.45, 0.34, 0.21)
def copy_samples(input_root, output_root, total_samples=10, ratio=(0.45, 0.34, 0.21)):
    # ... (保持你原来的代码不变) ...
    images_input_root = os.path.join(input_root, 'images', 'training')
    labels_input_root = os.path.join(input_root, 'annotations', 'training')

    images_output_root = os.path.join(output_root, 'images', 'training')
    labels_output_root = os.path.join(output_root, 'annotations', 'training')

    if os.path.exists(images_output_root):
        clear_directory(images_output_root)
    else:
        os.makedirs(images_output_root)

    if os.path.exists(labels_output_root):
        clear_directory(labels_output_root)
    else:
        os.makedirs(labels_output_root)

    quality_folders = ['high_quality', 'medium_quality', 'low_quality']

    total_high = int(total_samples * ratio[0])
    total_medium = int(total_samples * ratio[1])
    total_low = total_samples - total_high - total_medium

    all_selected_images = []

    for quality, num_samples in zip(quality_folders, [total_high, total_medium, total_low]):
        image_files = [f for f in os.listdir(os.path.join(images_input_root, quality)) if f.endswith('.png')]
        random.shuffle(image_files)

        selected_images = image_files[:num_samples]
        all_selected_images.extend([(quality, img) for img in selected_images])

    for quality, image_name in all_selected_images:
        image_path = os.path.join(images_input_root, quality, image_name)
        label_name = image_name.replace('.png', '_1stHO.png')
        label_path = os.path.join(labels_input_root, quality, label_name)

        if os.path.exists(label_path):
            shutil.copy(image_path, os.path.join(images_output_root, image_name))
            shutil.copy(label_path, os.path.join(labels_output_root, label_name))
        else:
            print(f"Warning: Label not found for {image_name}. Skipping.")
    print(f"Successfully copied {len(all_selected_images)} image and label pairs!")


# === 新增执行入口 ===
if __name__ == "__main__":
    import argparse
    import math
    parser = argparse.ArgumentParser(description='Sample segmentation image/label pairs by quality ratio')
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--total-samples', type=int, default=373)
    parser.add_argument('--ratio', type=float, nargs=3, default=(0.45, 0.34, 0.21))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.total_samples <= 0 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in args.ratio) or not math.isclose(sum(args.ratio), 1, abs_tol=1e-8):
        parser.error('sample count must be positive and ratios must be in [0,1] and sum to 1')
    if os.path.normcase(os.path.abspath(args.input_root)) == os.path.normcase(os.path.abspath(args.output_root)):
        parser.error('input-root and output-root must be different')
    random.seed(args.seed)
    copy_samples(args.input_root, args.output_root, args.total_samples, tuple(args.ratio))
