# mmcv_custom/runner/optimizer.py

import torch
from mmcv.runner import OptimizerHook, HOOKS

@HOOKS.register_module()
class DistOptimizerHook(OptimizerHook):
    """Optimizer hook using PyTorch native AMP (no Apex)."""

    def __init__(self, update_interval=1, grad_clip=None, coalesce=True, bucket_size_mb=-1, use_fp16=False):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16
        self.scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        runner.outputs['loss'] /= self.update_interval

        if self.use_fp16:
            # Native PyTorch autocast
            with torch.cuda.amp.autocast():
                loss = runner.outputs['loss']
            self.scaler.scale(loss).backward()
        else:
            runner.outputs['loss'].backward()

        if self.every_n_iters(runner, self.update_interval):
            if self.use_fp16:
                if self.grad_clip is not None:
                    self.scaler.unscale_(runner.optimizer)
                    self.clip_grads(runner.model.parameters())
                self.scaler.step(runner.optimizer)
                self.scaler.update()
            else:
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())
                runner.optimizer.step()

            runner.optimizer.zero_grad()
