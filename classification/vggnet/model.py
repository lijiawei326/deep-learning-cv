import torch
import torch.nn as nn

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'M'],
    'vgg16': [64, 64, 'M', 128, 128, "M", 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,'M']
}


class VGG(nn.Module):
    def __init__(self, model_name='vgg16', init_weights=False):
        super(VGG, self).__init__()
        self.cfg = cfgs[model_name]
        self.features = self.make_features()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 2)
        )
        if init_weights:
            self.initialize_weights_()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def make_features(self):
        layers = []
        in_channels = 3
        for m in self.cfg:
            if m == 'M':
                layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
            else:
                layers.extend([nn.Conv2d(in_channels, m, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True)])
                in_channels = m
        return nn.Sequential(*layers)

    def initialize_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


# net = VGG(model_name='vgg11')
# a = torch.rand(1, 3, 224, 224)
# y = net(a)
# print(y.shape)
