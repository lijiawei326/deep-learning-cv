# 各图像分类模型Pytorch复现--以Kaggle猫狗数据集为例

## 数据集介绍
train文件夹中包含25,000张猫和狗的图片。该文件夹中每张图片的文件名包含了该图片的序号和标签。\
test文件夹中包含12,000张包含id的无标签图像。我们需要预测test文件夹中所有图片是狗的概率，并上传至Kaggle对预测计算LogLoss进行评估。

## 文件结构
```
├── data : 存放图片数据的文件夹
│   ├── test
│   └── train
├── mean_std.py : 自动计算数据集均值与标准差
├── model.py
├── my_dataset.py
├── predict.py
├── split_data.py : 自动随机分割测试集与验证集
└── train.py
```

## 注意事项
* mean_std.py中以batch_size = 1 与batch_size = 25000计算得到的标准差由于计算机精度问题有些许差异。并且是否进行Resize也会影响标准差大小。但是计算出的均值是相等的。
* 由于Kaggle提交的submission的要求，predict.py输出的是每张图片为狗的概率。

## 各模型预测结果得分

* AlexNet : 0.64348
* VGG16-from scratch : 0.19161
* VGG16-Fine-Tuning : 0.07444
