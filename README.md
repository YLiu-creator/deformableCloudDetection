# Deformable Convolutional Cloud Detection Network - DCNet 


This is a reference implementation of DCNet in PyTorch. DCNet is designed to have an encoder-decoder architecture with the
deformable convolution mechanism. The encoder converts input images into a low dimensional space, thus robust category information is more easily captured in the deeper level. The decoder is designed to perform the inverse process of encoding, where cloud details and spatial resolutions are gradually recovered.

Specific implementation details and references are being supplemented...

## Ablation architectures and the state-of-the-art models

DCNet does not need to load any pre-training weights. 
L1-L3 indicates whether the DCB is adopted symmetrically at the first, second, and third layers in the DCNet.

|  DCNet| state-of-the-art models |
|:--:|:--:|
| self_contrast   | FCN |
| DCB+L1          | U-Net |
| DCB+L12         | SegNet |
| DCB+L123 		  | CloudSegNet|
|                 | CloudU-Net|


## Datasets

 - [GF-1 WFV Cloud and Cloud Shadow Cover Validation Data](http://sendimage.whu.edu.cn/en/mfc-validation-data/)


## Train
Run train_DCNet.py to train your model on GF-1 WFV datasets. You can also parallel your training on multi GPUs with '--gpu_id 0,1,2,3...' .

## Test
Run test_DCNet.py to evaluate. Results will be saved at ./results.
