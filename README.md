# SiamFC-RBA

## 1 Introduction
This is the code for improved Fully Convolutional Siamese tracker based on Response Behavior analysis. 

This code is based on  https://github.com/huanglianghua/siamfc-pytorch and improved by adding a Response Behavior analysis module.
 
## 2 Installation
Install Anaconda and then
```
# install PyTorch >= 1.0
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
# intall OpenCV using menpo channel (otherwise the read data could be inaccurate)
conda install -c menpo opencv
# install GOT-10k toolkit
pip install got10k
```

## 3 Run the demo
1.Setup the sequence path in tools/demo.py.

2.Setup the checkpoint path of your pretrained model. The Default model is located in tools/pretrained/
