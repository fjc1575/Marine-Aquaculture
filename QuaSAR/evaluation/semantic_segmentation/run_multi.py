import cv2
import argparse
import os
import mmcv
import torch

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv_custom import train_segmentor
from data_paths import apply_data_root

def parse_args(custom_args=None):
    parser = argparse.ArgumentParser(description='mmseg test (and save) a model')
    parser.add_argument('--config', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'configs', 'upernet',
        'vit_small_512_ade20k_160k.py'), help='test config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--data-root')
    parser.add_argument('--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--gpu-collect', action='store_true', help='use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp directory for collecting results')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config options')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    if custom_args:
        return parser.parse_args(custom_args)
    else:
        return parser.parse_args()


def main(custom_args=None, folder_name=None, config_overrides=None):
    args = parse_args(custom_args)

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if config_overrides:
        cfg.merge_from_dict(config_overrides)
    if args.data_root is not None:
        apply_data_root(cfg, args.data_root, 'segmentation')
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed, shuffle=False)

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'PALETTE' not in checkpoint['meta']:
        checkpoint['meta']['PALETTE'] = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0 and args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)


def run_batch():
    parser = argparse.ArgumentParser(description='Run segmentation inference for checkpoint subdirectories')
    parser.add_argument('--checkpoint-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--checkpoint-name', default='iter_8000.pth')
    parser.add_argument('--data-root')
    parser.add_argument('--config', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'configs', 'upernet',
        'vit_small_512_ade20k_160k.py'))
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    for folder_name in sorted(os.listdir(args.checkpoint_root)):
        folder_path = os.path.join(args.checkpoint_root, folder_name)
        if os.path.isdir(folder_path):
            checkpoint_path = os.path.join(folder_path, args.checkpoint_name)
            if not os.path.isfile(checkpoint_path):
                print(f"Skipping {folder_name}: checkpoint not found: {args.checkpoint_name}")
                continue
            show_dir = os.path.join(args.output_root, folder_name)
            os.makedirs(show_dir, exist_ok=True)

            print(f'\n==== Running on: {folder_name} ====')
            custom_args = [
                '--config', args.config,
                '--checkpoint', checkpoint_path,
                '--show-dir', show_dir,
            ]
            if args.show:
                custom_args.append('--show')
            if args.data_root:
                custom_args.extend(['--data-root', args.data_root])
            main(custom_args, folder_name, args.cfg_options)


if __name__ == '__main__':
    run_batch()
