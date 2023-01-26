# VGG16模型复现

## Training
* 运用RandomResizedCrop默认scale=(0.08,1.0)模型val_loss在0.69附近浮动，考虑到可能是图片裁剪的问题，将scale调整为(0.8,1.0),结果收敛。
