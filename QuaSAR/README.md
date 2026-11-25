# QuaSAR

Code for **QuaSAR: A Quality-Aware Selection and Proportioning Framework for Self-Supervised Deep Learning in Marine SAR Imagery**.

This repository includes Gamma distribution fitting, data quality grouping, and downstream model training. The Automated Classification Pipeline is implemented using MATLAB's machine learning tools. Upstream self-supervised pretraining uses [DINO](https://github.com/facebookresearch/dino).

The workflow is:

1. Stretch SAR images by matching a reference Gamma distribution and extract image quality metrics.
2. Choose metric columns, calculate a composite quality score, and assign high, medium, and low quality groups.
3. Organize the groups into the data layouts below and sample different quality proportions.
4. Use DINO externally for self-supervised pretraining, and the downstream scripts here for floating-raft segmentation or ship detection.
5. Collect performance measurements for each proportion and fit the performance model to analyze recommended proportions and marginal effects.

## Repository layout

```text
.
├── quality/
│   ├── quality_pipeline.py         # Gamma stretching and metric extraction
│   └── quality_scoring.py          # PCA-based scoring and three-group clustering
├── evaluation/
│   ├── semantic_segmentation/
│   │   ├── train.py                # Train a segmentation model
│   │   ├── test.py                 # Evaluate or visualize a checkpoint
│   │   ├── proportion.py           # Sample proportions and train each group
│   │   ├── data_sampler.py         # Sampling only; includes generate_ratios
│   │   ├── run_multi.py            # Run inference for multiple checkpoints
│   │   ├── configs/                # UPerNet/ViT-small and base configurations
│   │   └── mmcv_custom/            # Backbone registration, optimizer, Apex runner
│   └── object_detection/
│       ├── prepare_ship_ratios.py  # Sample ships and convert YOLO labels to COCO
│       ├── train.py                # Train a ship detector
│       ├── test.py                 # Evaluate or visualize a detector
│       ├── configs/                # Cascade R-CNN/ViT-small configurations
│       └── mmcv_custom/            # Detector backbone and training extensions
├── analysis/
│   └── performance_analysis.py     # Quadratic modeling, optimization, marginal effects
├── models/                        # Shared Vision Transformer implementation
├── mmseg/                         # Bundled MMSegmentation 0.12.0 source
└── data_paths.py                   # Shared dataset-root configuration helper
```
