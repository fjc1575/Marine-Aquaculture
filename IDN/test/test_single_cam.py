import warnings

import torchvision
from CAM.grad_cam_testing import GradCAM
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
import torch.nn.functional as F
import os
from pathlib import Path
import glob

from metric.explainable_metric import InsertionIoU, DeletionIoU, compute_iou
from parameters import setupArgs
from train.train_bpcamnet import Supervision_Train
from utils.cuda import setupCuda
from utils.seed import setSeed
import time
import pandas as pd
import time
import csv



def create_dir(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def save_results_to_csv(results, file_path):
    """Save results to a CSV file."""
    with open(file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        header = ["Metric", "AUC"] + [f"Step_{i+1}" for i in range(len(results[0]["Scores"]))]
        writer.writerow(header)

        # Write rows
        for result in results:
            row = [result["Metric"], float(result["AUC"])] + [float(score) for score in result["Scores"]]
            writer.writerow(row)

    print(f"Saved results to {file_path}")


def process_single_image(image_path, label_path, output_folder, model, target_layer):
    """Process a single image and save results."""
    # Initialize metrics
    insertion = InsertionIoU(model=model, n_steps=50, batch_size=10, baseline="black")
    deletion = DeletionIoU(model=model, n_steps=50, batch_size=10, baseline="black")

    results = []

    # Load and preprocess image and label
    image_name = os.path.basename(image_path).split('.')[0]  # Extract the image name without extension
    rgb_image = Image.open(image_path)
    label_image = Image.open(label_path)

    rgb_numpy = np.array(rgb_image) / 255.0
    rgb_image = torch.tensor(rgb_numpy).unsqueeze(0).permute(0, 3, 1, 2)
    label_image = torchvision.transforms.ToTensor()(label_image).to('cuda')

    # Model forward pass
    output = model(rgb_image.float().to('cuda'))
    mask = torch.argmax(F.softmax(output, dim=1), dim=1).cpu().numpy()

    # Define target class
    targets = [SemanticSegmentationTarget(0, mask[0])]

    # Calculate CAM
    start_time = time.time()
    with GradCAM(model=model,
                  target_layers=target_layer,
                  use_cuda=torch.cuda.is_available(),
                  pre_mask = mask
                 ) as cam:
        grayscale_cam = cam(input_tensor=rgb_image.float().to('cuda'), targets=targets)[0]
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

    # Compute metrics
    IoU = compute_iou(torch.tensor(mask).cuda().to(torch.int), label_image.to(torch.int))
    insertion_auc, insertion_scores = insertion(rgb_image.float().to('cuda'), torch.tensor(grayscale_cam).cuda(), label_image.to(torch.int))
    deletion_auc, deletion_scores = deletion(rgb_image.float().to('cuda'), torch.tensor(grayscale_cam).cuda(), label_image.to(torch.int))

    insertion_auc = float(insertion_auc) / IoU
    deletion_auc = float(deletion_auc) / IoU

    # Append results for saving
    results.append({
        'Metric': 'Insertion',
        'AUC': insertion_auc,
        'Scores': insertion_scores / IoU.tolist()  # Ensure tensor is converted to list
    })
    results.append({
        'Metric': 'Deletion',
        'AUC': deletion_auc,
        'Scores': deletion_scores / IoU.tolist()  # Ensure tensor is converted to list
    })

    # Save CAM image
    cam_image = show_cam_on_image(rgb_numpy, grayscale_cam, use_rgb=True, image_weight=0.5)
    cam_image = Image.fromarray(cam_image)
    cam_image_path = os.path.join(output_folder, f"{image_name}_cam.png")
    cam_image.save(cam_image_path)
    print(f"Saved CAM for {image_name} at {cam_image_path}")

    # Save results to CSV
    csv_path = os.path.join(output_folder, f"{image_name}_results.csv")
    save_results_to_csv(results, csv_path)

    print(f"Processing time: {processing_time_ms:.2f} ms")
    print(f"Insertion AUC: {insertion_auc}")
    print(f"Deletion AUC: {deletion_auc}")


def main():
    warnings.filterwarnings("ignore")

    # Initialize operations
    opt = setupArgs()
    setSeed(opt)
    setupCuda(opt)

    # Load model
    # model_lighting = Supervision_Train.load_from_checkpoint(r'C:\polsar\resource\MPA_RUet_Source\model\lfi0.5_epoch150\lfi0.5.ckpt', opt=opt)
    model_lighting = Supervision_Train.load_from_checkpoint(r'bpcamnet-4444-gf3.ckpt', opt=opt)

    model_lighting.eval()
    model = model_lighting.net

    target_layer = [
        model.backbone.layer4[-1],
    ]

    # Define paths
    image_path = r'C:\polsar\datasets\高分3polsar数据_wishart\lee_dataset\test\image\2 (48).png'
    label_path = r'C:\polsar\datasets\高分3polsar数据_wishart\lee_dataset\test\label\2 (48).png'
    output_folder = r'C:\polsar\resource\MPA_RUet_Source\test_cam_result\insertion_deletion\with'
    create_dir(output_folder)

    # Process single image
    process_single_image(image_path, label_path, output_folder, model, target_layer)


if __name__ == "__main__":
    main()