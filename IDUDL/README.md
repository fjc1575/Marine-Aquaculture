# IDUDL: Incremental Double Unsupervised Deep Learning Model

The Paper is under review


## Code files

**IDUDL** is defined as **FCSSN** and **FEN** in model.py.

main.py defines the training logic. It import all .py files except demo.py.

demo.py provides an example.

## How to run

###  main()
Running main.py can train and test a model with no labels. You can just train a model or test a model by denoting some codes. There are some output details.

In the output path, seven folders are generated in each round. For example, in the first round, they are:

**1**: the FEN output pseudo-lablels

**1FCSSNout**: the FCSSN output aquauclture extraction results

**1FCSSNpth**: the FCSSN parameters  file

**1trainimg**: (the training set) input of FCSSN (input image)

**1trainlabel**: (the training set) pseudo-labels corresponding to FCSSN input image from **1** (the FEN output pseudo-labels)

**1valimg**: (the val set)  input of FCSSN (input image)

**1vallabel**: (the val set) pseudo-labels corresponding to FCSSN input image from **1** (the FEN output pseudo-labels)

Retain the intermediate process, which can be adjusted according to needs at any time.



### demo()

We provide a training convergence result. You can run demo.py

The demo images in **'./IDUDL/data/demo'**.

The weight file in **'./IDUDL/pth'**.



IDUDL.pth is uploaded to Link: https://pan.baidu.com/s/1JvtBkzEb3OexiZjynmaJNQ  code: q5k3 
