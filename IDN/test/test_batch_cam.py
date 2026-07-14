import warnings

import torchvision
# from pytorch_grad_cam import XGradCAM, GradCAMPlusPlus, LayerCAM, FullGrad, HiResCAM, \
#     GradCAMElementWise, AblationCAM, ScoreCAM, EigenCAM,GradCAM
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
import torch.nn.functional as F
import os
from pathlib import Path
import glob

from CAM.grad_cam_testing import GradCAM
from metric.explainable_metric import InsertionIoU, DeletionIoU, compute_iou
from parameters import setupArgs
from train.train_bpcamnet import Supervision_Train
from utils.cuda import setupCuda
from utils.seed import setSeed
import time

# 创建指定的保存目录
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 读取并处理文件夹中的每一张图片
def process_images_in_folder(image_folder,label_folder, output_folder, model, target_layer):

    insertion = InsertionIoU(model=model, n_steps=50, batch_size=10, baseline="black")
    deletion = DeletionIoU(model=model, n_steps=50, batch_size=10, baseline="black")

    insetrtion_list = []
    deletion_list = []
    iou_list = []
    total_processing_time = 0  # 初始化总处理时间
    num_images = 0  # 初始化图片计数

    # 获取所有png文件
    image_paths = glob.glob(os.path.join(image_folder, '*.png'))
    label_paths = glob.glob(os.path.join(label_folder, '*.png'))

    for image_path,label_path in zip(image_paths,label_paths):

        num_images += 1

        # 读取图片并进行预处理
        image_name = os.path.basename(image_path)  # 提取图片名称

        rgb_image = Image.open(image_path)
        label_image = Image.open(label_path)

        rgb_numpy = np.array(rgb_image) / 255.0

        rgb_image = torch.tensor(rgb_numpy).unsqueeze(0).permute(0, 3, 1, 2)
        label_image = torchvision.transforms.ToTensor()(label_image).to('cuda')

        # 模型前向传播
        output = model(rgb_image.float().to('cuda'))
        mask = torch.argmax(F.softmax(output, dim=1), dim=1).cpu().numpy()



        # 定义目标类
        targets = [SemanticSegmentationTarget(0, mask[0])]

        # 使用XGradCAM计算类激活图
        # GradCAM,GradCAMPlusPlus,HiResCAM, LayerCAM, XGradCAM,ScoreCAM,AblationCAM,EigenCAM
        start_time = time.time()
        with GradCAM(model=model,
                    target_layers=target_layer,
                    use_cuda=torch.cuda.is_available(),
                    # model_output=output,
                    pre_mask=mask

                    ) as cam:
            grayscale_cam = cam(input_tensor=rgb_image.float().to('cuda'), targets=targets)[0]
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000  # 转换为毫秒
            total_processing_time += processing_time_ms  # 累加每次处理时间


            IoU = compute_iou(torch.tensor(mask).cuda().to(torch.int),label_image.to(torch.int))

            insertion_auc, insertion_scores = insertion(rgb_image.float().to('cuda'), torch.tensor(grayscale_cam).cuda(), label_image.to(torch.int))
            deletion_auc, deletion_scores = deletion(rgb_image.float().to('cuda'), torch.tensor(grayscale_cam).cuda(), label_image.to(torch.int))

            insertion_auc , insertion_scores = insertion_auc / IoU , insertion_scores / IoU
            deletion_auc , deletion_scores = deletion_auc / IoU, deletion_scores / IoU

            insetrtion_list.append(float(insertion_auc))
            deletion_list.append(float(deletion_auc))
            iou_list.append((float(IoU)))

            cam_image = show_cam_on_image(rgb_numpy, grayscale_cam, use_rgb=True, image_weight=0.0)
            cam_image = Image.fromarray(cam_image)

            # 保存类激活图
            output_image_path = os.path.join(output_folder, image_name)
            cam_image.save(output_image_path)
            print(f"Saved CAM for {image_name} at {output_image_path}")

    avg_processing_time_ms = total_processing_time / num_images if num_images > 0 else 0
    print(f"Average processing time per image: {avg_processing_time_ms:.2f} ms")



    return sum(insetrtion_list)/len(insetrtion_list) , sum(deletion_list)/len(deletion_list) , sum(insetrtion_list)/len(insetrtion_list) - sum(deletion_list)/len(deletion_list)



# 主程序
def main():
    warnings.filterwarnings("ignore")

    # 初始化操作
    opt = setupArgs()
    setSeed(opt)
    setupCuda(opt)

    # model_lighting = Supervision_Train.load_from_checkpoint(r'C:\polsar\code\PolSARSeg - 副本\weights\bpcamnet-4444-gf3.ckpt', opt=opt)
    model_lighting = Supervision_Train.load_from_checkpoint(r'C:\polsar\resource\MPA_RUet_Source\model\lfi0.5_epoch150\lfi0.5.ckpt', opt=opt)
    model_lighting.eval()
    model = model_lighting.net

    target_layer = [
        model.backbone.layer4[-1],
    ]

    # 定义输入和输出文件夹
    image_folder = r'C:\polsar\datasets\高分3号Polsar数据\aug\test\image'  # 输入图片的文件夹路径
    label_folder = r'C:\polsar\datasets\高分3号Polsar数据\aug\test\label'
    output_folder = r'C:\polsar\resource\MPA_RUet_Source\test_cam_result\pscam'  # 输出类激活图的文件夹路径
    create_dir(output_folder)

    # 处理输入文件夹中的所有图片
    insetrion,deletion,overall = process_images_in_folder(image_folder,label_folder, output_folder, model, target_layer)
    print('insetrion: ',insetrion )
    print('deletion' , deletion)
    print('overall', overall)

if __name__ == "__main__":
    main()
