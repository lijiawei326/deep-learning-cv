# GoogLeNet模型复现
## aux参数为布尔值，但是传入时总识别为True【已解决】
由于输入被默认为字符串，因此输入任何值都为True。若不需要使用辅助分类器，不传入aux参数即可。

## train_loss未收敛时val_loss达到最低，随后上升至收敛【未解决】
猜测为模型过拟合。

![loss](./googlenet.png)
