import torch
import torch.nn as nn


class InceptionV3(nn.Module):
    def __init__(self, aux_logits=False,init_weights=False):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.conventional_net = nn.Sequential(
            # 3 x 299 x 299
            BasicConv2d(3, 32, kernel_size=(3, 3), stride=(2, 2)),
            # 32 x 149 x 149
            BasicConv2d(32, 32, kernel_size=(3, 3)),
            # 32 x 147 x147
            BasicConv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            # 64 x 147 x 147
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            # 64 x 73 x 73
            BasicConv2d(64, 80, kernel_size=(3, 3)),
            # 80 x 71 x 71
            BasicConv2d(80, 192, kernel_size=(3, 3), stride=(2, 2)),
            # 192 x 35 x 35
            BasicConv2d(192, 288, kernel_size=(3, 3), padding=(1, 1))
        )
        # 288 x 35 x 35
        self.Inception_aux = nn.Sequential(
            InceptionA(288, [64, 96], [48, 64], 64, 64),  # 3a
            InceptionA(288, [64, 96], [48, 64], 64, 64),  # 3b
            InceptionA(288, [64, 96], [128, 384], 0, 0),  # 3c
            # 768 x 17 x 17
            InceptionB(768, [128, 192], [128, 192], 192, 192),  # 4a
            InceptionB(768, [160, 192], [160, 192], 192, 192),  # 4b
            InceptionB(768, [160, 192], [160, 192], 192, 192),  # 4c
            InceptionB(768, [192, 192], [192, 192], 192, 192),  # 4d
        )
        # 768 x 17 x 17
        self.aux_Inception = nn.Sequential(
            InceptionB(768, [192, 192], [192, 320], 0, 0),  # 4e
            # 1280 x 8 x 8
            InceptionC(1280, [448, 384], 384, 192, 320),  # 5a
            InceptionC(2048, [448, 384], 384, 192, 320),  # 5b
        )
        # 2048 x 8 x 8
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(2048,2)
        )
        self.aux = Aux()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conventional_net(x)
        x = self.Inception_aux(x)
        if self.aux_logits and self.training:
            aux = self.aux(x)
        x = self.aux_Inception(x)
        x = self.classifier(x)
        if self.aux_logits and self.training:
            return x, aux
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class InceptionA(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionA, self).__init__()
        self.reduce = True if c4 == 0 else False
        if not self.reduce:
            stride = (1, 1)
            padding = (1, 1)
            self.branch3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                BasicConv2d(in_channels, c3, kernel_size=(1, 1))
            )
            self.branch4 = BasicConv2d(in_channels, c4, kernel_size=(1, 1))
        else:
            stride = (2, 2)
            padding = (0, 0)
            self.branch3 = nn.MaxPool2d(kernel_size=(3, 3), stride=stride)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, c1[0], kernel_size=(1, 1)),
            BasicConv2d(c1[0], c1[1], kernel_size=(3, 3), padding=(1, 1)),
            BasicConv2d(c1[1], c1[1], kernel_size=(3, 3), stride=stride, padding=padding)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, c2[0], kernel_size=(1, 1)),
            BasicConv2d(c2[0], c2[1], kernel_size=(3, 3), stride=stride, padding=padding)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if not self.reduce:
            x4 = self.branch4(x)
            return torch.cat([x1, x2, x3, x4], dim=1)
        else:
            return torch.cat([x1, x2, x3], dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionB, self).__init__()
        self.reduce = True if c4 == 0 else False
        if not self.reduce:
            padding = 3
            stride = 1
            self.branch3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                BasicConv2d(in_channels, c3, kernel_size=(1, 1))
            )
            self.branch4 = BasicConv2d(in_channels, c4, kernel_size=(1, 1))
        else:
            padding = 2
            stride = 2
            self.branch3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, c1[0], kernel_size=(1, 1)),
            Separable_Conv2d(c1[0], c1[1], padding=3),
            Separable_Conv2d(c1[1], c1[1], padding=padding, stride=stride)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, c2[0], kernel_size=(1, 1)),
            Separable_Conv2d(c2[0], c2[1], stride=stride, padding=padding)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if not self.reduce:
            x4 = self.branch4(x)
            return torch.cat([x1, x2, x3, x4], dim=1)
        else:
            return torch.cat([x1, x2, x3], dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionC, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, c1[0], kernel_size=(1, 1)),
            BasicConv2d(c1[0], c1[1], kernel_size=(3, 3), padding=(1, 1)),
            Concat_Separable_Conv2d(c1[1])
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, c2, kernel_size=(1, 1)),
            Concat_Separable_Conv2d(c2)
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            BasicConv2d(in_channels, c3, kernel_size=(1, 1))
        )
        self.branch4 = BasicConv2d(in_channels, c4, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class Separable_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, stride=1):
        super(Separable_Conv2d, self).__init__()
        self.separable_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), stride=(1, stride), padding=(0, padding)),
            nn.Conv2d(in_channels, out_channels, kernel_size=(7, 1), stride=(stride, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.separable_conv(x)


class Concat_Separable_Conv2d(nn.Module):
    def __init__(self, in_channels):
        super(Concat_Separable_Conv2d, self).__init__()
        self.c1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1))
        self.c2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(in_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.bn(x))
        return x


class Aux(nn.Module):
    def __init__(self):
        super(Aux, self).__init__()
        self.aux = nn.Sequential(
            # 768 x 17 x 17
            nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3)),
            # 768 x 5 x 5
            BasicConv2d(768, 128, kernel_size=(1, 1)),
            # 128 x 5 x 5
            nn.Flatten(start_dim=1),
            # 3200
            nn.Linear(3200, 1024),
            nn.Linear(1024, 2)
        )

    def forward(self,x):
        return self.aux(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.basic(x)


# net = InceptionV3(aux_logits=True)
# a = torch.rand((1, 3, 299, 299))
# b,aux = net(a)
# print(b.shape,aux.shape)
# b = net(a)
# print(b.shape)
