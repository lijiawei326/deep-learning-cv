import torch
import torch.nn as nn


def nin_block(in_channels, out_channels, kernal_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, stride=stride, padding=padding, kernel_size=kernal_size),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )


class NiN(nn.Module):
    def __init__(self, init_weights=False):
        super(NiN, self).__init__()
        self.nin = nn.Sequential(
            nin_block(3, 96, kernal_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nin_block(96, 256, kernal_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nin_block(256, 384, kernal_size=3, stride=1, padding=1),
            nin_block(384, 2, kernal_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(start_dim=1)
        )

        if init_weights:
            self.initialize_weights_()

    def forward(self, x):
        return self.nin(x)

    def initialize_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


net = NiN(init_weights=True)
# for m in net.modules():
#     print(m.state_dict())
a = torch.rand((3, 3, 224, 224))
y = net(a)
print(y)
