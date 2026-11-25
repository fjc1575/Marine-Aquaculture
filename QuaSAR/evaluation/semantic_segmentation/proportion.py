from data_sampler import copy_samples,generate_ratios
import importlib
import train
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from mmsegmentation library:
https://github.com/open-mmlab/mmsegmentation/blob/master/tools/train.py
"""
import sys
import argparse
import copy
import os
import os.path as osp
import json
import math
import random
import time
import mmcv
import torch
import warnings
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import set_random_seed #, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
from mmcv_custom import train_segmentor

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', default=osp.join(
        osp.dirname(osp.abspath(__file__)), 'configs', 'upernet',
        'vit_small_512_ade20k_160k.py'))
    parser.add_argument('--input-root', required=True,
                        help='dataset root with images/training/{high_quality,medium_quality,low_quality}')
    parser.add_argument('--work-dir', required=True,
                        help='parent output directory; each ratio gets its own subdirectory')
    parser.add_argument('--total-samples', type=int, default=200)
    parser.add_argument('--ratio', type=float, nargs=3, action='append',
                        metavar=('HIGH', 'MEDIUM', 'LOW'),
                        help='ratio to run; repeat for multiple ratios; omit for all 66 ratios')
    parser.add_argument('--validation-root',
                        help='validation/test dataset root; defaults to input-root')
    parser.add_argument(
        '--load-from', help='checkpoint compatible with the segmentation model')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--fast-eval',
        action='store_true',
        help='whether to perform fast evaluation.')
    args = parser.parse_args()
    if args.total_samples <= 0:
        parser.error('--total-samples must be positive')
    if args.ratio is not None and any(
            any(not math.isfinite(v) or v < 0 or v > 1 for v in ratio)
            or not math.isclose(sum(ratio), 1.0, abs_tol=1e-8)
            for ratio in args.ratio):
        parser.error('--ratio values must be between 0 and 1 and sum to 1')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def set_dataset_root(dataset_cfg, data_root):
    """Set the root through RepeatDataset/ConcatDataset wrappers."""
    if 'dataset' in dataset_cfg:
        set_dataset_root(dataset_cfg['dataset'], data_root)
    elif 'datasets' in dataset_cfg:
        for child in dataset_cfg['datasets']:
            set_dataset_root(child, data_root)
    else:
        dataset_cfg['data_root'] = data_root


def main():
    args = parse_args()
    input_root = osp.abspath(args.input_root).replace('\\', '/')
    validation_root = osp.abspath(args.validation_root or args.input_root).replace('\\', '/')

    # 遍历所有有效比例并调用 copy_samples
    ratios = [tuple(ratio) for ratio in args.ratio] if args.ratio is not None else generate_ratios(
        start=0.0, end=1.0, step=0.1)
    for idx, ratio in enumerate(ratios):
        # 根据当前比例生成输出路径
        work_dir = os.path.join(
            osp.abspath(args.work_dir),
            f'output_high_{ratio[0]}_mid_{ratio[1]}_low_{ratio[2]}'
        ).replace('\\', '/')

        completed_path = osp.join(work_dir, 'training_complete.json')
        if os.path.isfile(completed_path):
            print(
                f"[{idx + 1}/{len(ratios)}] 跳过已完成比例: 高质量={ratio[0]} 中质量={ratio[1]} 低质量={ratio[2]}")
            continue
        if osp.exists(work_dir):
            raise FileExistsError(
                f'Output exists without a completion record: {work_dir}. '
                'Choose a new --work-dir to preserve this run.')

        print(f"[{idx + 1}/{len(ratios)}] 当前比例: 高质量={ratio[0]} 中质量={ratio[1]} 低质量={ratio[2]}")

        cfg = Config.fromfile(args.config)
        if args.options is not None:
            cfg.merge_from_dict(args.options)
        if args.load_from is not None:
            cfg.load_from = args.load_from
        if cfg.get('load_from') and not osp.isfile(cfg.load_from):
            raise FileNotFoundError(cfg.load_from)

        # Each training dataset must read the samples selected for this ratio.
        output_root = osp.join(work_dir, 'sampled_data').replace('\\', '/')
        if args.seed is not None:
            random.seed(args.seed)
        copy_samples(input_root, output_root, total_samples=args.total_samples, ratio=ratio)
        selected = sorted(os.listdir(osp.join(output_root, 'images', 'training')))
        if len(selected) != args.total_samples:
            raise RuntimeError(f'Expected {args.total_samples} samples, copied {len(selected)}')
        set_dataset_root(cfg.data.train, output_root)
        for split in ('val', 'test'):
            if split in cfg.data:
                set_dataset_root(cfg.data[split], validation_root)
        cfg.data_root = output_root
        with open(osp.join(work_dir, 'sampling.json'), 'w', encoding='utf-8') as handle:
            json.dump(dict(input_root=input_root, training_root=output_root,
                           validation_root=validation_root, ratio=ratio,
                           total_samples=args.total_samples, seed=args.seed,
                           images=selected), handle, ensure_ascii=False, indent=2)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # 更新工作目录
        cfg.work_dir = work_dir
        if args.load_from is not None:
            cfg.load_from = args.load_from
        if args.resume_from is not None:
            cfg.resume_from = args.resume_from
        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids
        else:
            cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

        # 初始化分布式训练
        distributed = args.launcher != 'none'
        if distributed:
            init_dist(args.launcher, **cfg.dist_params)

        # 创建工作目录
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # 保存配置文件
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

        # 初始化日志
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # 记录环境信息
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        logger.info(f'Environment info:\n{env_info}')

        # 设置随机种子
        if args.seed is not None:
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed

        # 构建模型
        model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        if not distributed:
            model = revert_sync_batchnorm(model)

        # 构建数据集
        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))

        # 设置模型类别信息
        model.CLASSES = datasets[0].CLASSES

        # 开始训练
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=dict(env_info=env_info, seed=args.seed)
        )
        with open(completed_path, 'w', encoding='utf-8') as handle:
            json.dump(dict(ratio=ratio, total_samples=args.total_samples,
                           max_iters=cfg.runner.max_iters), handle, indent=2)


if __name__ == '__main__':
    main()


