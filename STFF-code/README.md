# A Self-supervised Transformer with Feature Fusion for SAR Image Semantic Segmentation in Marine Aquaculture Monitoring

The Paper is under review


## Code files

**STFF** is consist of **Transformer encoding** and **Semantic segmentation decoding** in **self-distillation-transformer-encoder** and **feature-fusion-decoder**.

**self-distillation-transformer-encoder** includes the **main.py** to train the transformer encoding with self-distillation.

**feature-fusion-decoder** includes the **train.py** and **test.py** to train the semantic segmentation decoding and test the STFF.

**test.py** provides an example.

## How to run

###  main.py
Running main.py can train STFF with no labels. Need set the data_path and output_dir

In the output path (output_dir), checkpoint.pth will be obatined

###  trian.py
--config ..\feature-fusion-decoder\configs\upernet\vit_small_512_ade20k_160k_ms.py --work-dir ..\work_log --load-from ..\pth\checkpoint.pth --seed 0

###  test.py
--config ..\feature-fusion-decoder\configs\upernet\vit_small_512_ade20k_160k_ms.py --checkpoint ..\work_log\latest.pth --show-dir ..\work_log

### demo()

We provide a training convergence result. You can run test.py

The demo images in **'./data'**.

The weight file in **'./work_log/latest.pth'**.



## Pre-trained models

They are available at https://pan.baidu.com/s/1zakqEgxWkCqVHcATZ5XnAw code：viro 

