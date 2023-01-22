import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 左,右,上,下
            nn.ZeroPad2d((2, 1, 2, 1)),  # [3,224,224] -> [3,227,227]
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),  # [3,224,224] -> [96,55，55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),  # [96,55,55] -> [96,27,27]

            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2),  # [96,27,27] -> [256,27,27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),  # [256,27,27] -> [256,13,13]

            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1),  # [256,13,13] -> [384,13,13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1),  # [384,13,13] -> [384,13,13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),  # [384,13,13] -> [256,13,13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),  # [256,13,13] -> [256,6,6]
        )

        self.classifier = nn.Sequential(
            # 原文中是4096，为降低计算量
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


# model = AlexNet()
# model.eval()
# with torch.no_grad():
#     a = torch.rand((1,3,224,224))
#     a = model(a)
#     print(a.shape)
