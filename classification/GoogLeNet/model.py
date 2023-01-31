import torch
import torch.nn as nn


class GooLeNet(nn.Module):
    def __init__(self, aux_logits=False,init_weights=False):
        super(GooLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.Conventional_Net = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            BasicConv2d(64, 192, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        self.inception_aux1 = nn.Sequential(
            # inception 3a
            Incepiton(192, 64, (96, 128), (16, 32), 32),
            # inception 3b
            Incepiton(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            # inception 4a
            Incepiton(480, 192, (96, 208), (16, 48), 64)
        )
        self.aux1_aux2 = nn.Sequential(
            # inception 4b
            Incepiton(512, 160, (112, 224), (24, 64), 64),
            # inception 4c
            Incepiton(512, 128, (128, 256), (24, 64), 64),
            # inception 4d
            Incepiton(512, 112, (148, 288), (32, 64), 64),
        )
        self.aux2_inception = nn.Sequential(
            # inception 4e
            Incepiton(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            # inception 5a
            Incepiton(832, 256, (160, 320), (32, 128), 128),
            # inception 5b
            Incepiton(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024, 2)
        )
        if self.aux_logits:
            self.aux1 = Aux(512)
            self.aux2 = Aux(528)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.Conventional_Net(x)
        x = self.inception_aux1(x)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        x = self.aux1_aux2(x)
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        x = self.aux2_inception(x)
        x = self.classifier(x)
        if self.aux_logits and self.training:
            return x, aux1, aux2
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)


class Incepiton(nn.Module):
    def __init__(self, in_channel, c1, c2, c3, c4):
        super(Incepiton, self).__init__()
        self.branch1_1 = BasicConv2d(in_channel, c1, kernel_size=(1, 1))
        self.branch2_1 = BasicConv2d(in_channel, c2[0], kernel_size=(1, 1))
        self.branch2_2 = BasicConv2d(c2[0], c2[1], kernel_size=(3, 3), padding=(1, 1))
        self.branch3_1 = BasicConv2d(in_channel, c3[0], kernel_size=(1, 1))
        self.branch3_2 = BasicConv2d(c3[0], c3[1], kernel_size=(5, 5), padding=(2, 2))
        self.branch4_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.branch4_2 = BasicConv2d(in_channel, c4, kernel_size=(1, 1))

    def forward(self, x):
        output_1 = self.branch1_1(x)
        output_2 = self.branch2_2(self.branch2_1(x))
        output_3 = self.branch3_2(self.branch3_1(x))
        output_4 = self.branch4_2(self.branch4_1(x))
        return torch.cat((output_1, output_2, output_3, output_4), dim=1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Aux(nn.Module):
    def __init__(self, in_channels):
        super(Aux, self).__init__()
        self.aux = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3)),
            nn.Conv2d(in_channels, 128, kernel_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.7),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        return self.aux(x)


# net = GooLeNet(aux_logits=False)
# a = torch.rand((1, 3, 224, 224))
# net.train()
# a, b, c = net(a)
# print(a.shape)
# print(b.shape)
# print(c.shape)
# print(net.state_dict().keys())
