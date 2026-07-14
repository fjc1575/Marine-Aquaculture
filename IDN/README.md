# IDN

Official implementation for:

Jianchao Fan, Keyuan Liu, Danchen Zheng, Jun Wang*. "An Interpretable Dual-Branch Network With Consistent Learning for PolSAR Image Segmentation." IEEE Transactions on Aerospace and Electronic Systems, 2026, 62: 2694-2710.

## Overview

This repository contains the PyTorch/PyTorch Lightning implementation of IDN for PolSAR image segmentation. The code includes:

- Dual-branch segmentation models in `architectures/`
- Training scripts in `train/`
- Segmentation and CAM testing scripts in `test/`
- Dataset loading code in `datasets/`
- Metrics and losses in `metric/` and `loss/`

## Environment

The code was developed with Python 3.8, PyTorch 2.2.0, CUDA 11.8, and PyTorch Lightning 2.2.3.

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Note: `requirements.txt` contains several Windows local wheel paths for `torch`, `torchvision`, and some conda packages. If installation fails, install PyTorch and TorchVision manually first, for example:

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==2.2.3 lightning==2.2.3 albumentations==1.4.4 grad-cam==1.4.8 opencv-python pillow scipy scikit-image scikit-learn timm wandb
```

## Data and Pretrained Weights

Some datasets and testing weights are available from Baidu Netdisk:

- Link: https://pan.baidu.com/s/1kGolH4xh4IzWXjIWSgdqIg?pwd=vj2b
- Extraction code: `vj2b`
- Shared folder name: `IDN`

After downloading, organize the dataset with the following structure:

```text
dataset_root/
  train/
    image/
      *.png
    label/
      *.png
  test/
    image/
      *.png
    label/
      *.png
```

Image and label files must have matching file names. The current dataset loader reads three-channel Freeman decomposition images and binary segmentation labels.

## Configure Data Paths

The data paths are currently hard-coded in `datasets/seg_edge_dataset.py` inside `DataModule.__init__`.

Before training or testing, update these two variables:

```python
self.train_data_path = r"path\to\dataset_root\train"
self.validation_dataset_path = r"path\to\dataset_root\test"
```

If you use the augmented dataset, keep `--augmentation True`. If you use the Lee-filtered dataset, set `--augmentation False` and update the corresponding path branch in `DataModule.__init__`.

## Training

Run commands from the repository root.

Train the baseline dual-branch model:

```bash
python train/train_bpcamnet.py --gpu 0 --bs 8 --n_epochs 150 --lr 0.0003 --model_save_path path/to/save/checkpoints
```

Train the interpretable consistent-learning model:

```bash
python train/train_l1.py --gpu 0 --bs 8 --n_epochs 150 --lr 0.0003 --model_save_path path/to/save/checkpoints
```

`train/train_l1.py` iterates over the `gamma_list` defined at the bottom of the file:

```python
gamma_list = [0.01, 2, 5]
```

Modify this list if you want to train only one setting.

Checkpoints are saved under:

```text
<model_save_path>/<arch>_epoch<n_epochs>/
```

The checkpoint callback monitors `val_OA` and saves the best checkpoint as well as the last checkpoint.

## Testing Segmentation

To evaluate a saved checkpoint, edit the checkpoint path in `test/test_segmentation.py`:

```python
model = Supervision_Train.load_from_checkpoint(
    r"path\to\checkpoint.ckpt",
    opt=opt
)
```

Then run:

```bash
python test/test_segmentation.py --gpu 0
```

The script reports the following metrics:

- IoU
- F1
- OA
- Recall
- Kappa
- mIoU
- Precision

For `train/train_l1.py`, predicted masks are saved by `utils/save_image.py` to:

```text
<test_result_path>/<arch>/combined/
<test_result_path>/<arch>/mask/
```

You can set the output path with:

```bash
python test/test_segmentation.py --test_result_path path/to/test_results
```

## CAM Testing

For visualization and interpretability experiments, use:

```bash
python test/test_single_cam.py
python test/test_batch_cam.py
```

Before running these scripts, update the hard-coded checkpoint, image, label, and output paths near the bottom of each file.

## Important Arguments

Common command-line arguments are defined in `parameters/basic_training_params.py` and `parameters/wandb_parmas.py`.

| Argument | Default | Description |
| --- | --- | --- |
| `--gpu` | `[0]` | GPU IDs used by PyTorch Lightning |
| `--bs` | `8` | Batch size |
| `--n_epochs` | `150` | Number of training epochs |
| `--lr` | `0.0003` | Learning rate |
| `--num_class` | `2` | Number of segmentation classes |
| `--augmentation` | `True` | Selects the augmented training path branch |
| `--model_save_path` | hard-coded Windows path | Checkpoint output directory |
| `--test_result_path` | hard-coded Windows path | Test prediction output directory |
| `--log_online` | `False` | Enable Weights & Biases logging |

## Citation

If you use this code, please cite:

```bibtex
@article{fan2026idn,
  title={An Interpretable Dual-Branch Network With Consistent Learning for PolSAR Image Segmentation},
  author={Fan, Jianchao and Liu, Keyuan and Zheng, Danchen and Wang, Jun},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  volume={62},
  pages={2694--2710},
  year={2026}
}
```
