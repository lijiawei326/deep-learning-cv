import torch
from model import AlexNet
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from my_dataset import test_set
import pandas as pd
import torch.nn as nn


def main():
    # 定义设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}!")

    # 载入数据
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    test_dataset = test_set('./data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 预测
    weights_path = './alexnet.pth'
    assert os.path.exists(weights_path), f'{weights_path} does not exist!'

    weights_dict = torch.load(weights_path, map_location='cpu')

    alexnet = AlexNet()
    alexnet.load_state_dict(weights_dict)
    alexnet.to(device)
    alexnet.eval()
    ids = []
    labels = []
    softmax = nn.Softmax(dim=1)
    for data in test_loader:
        image, id = data
        ids.extend(id.tolist())
        image = image.to(device)
        with torch.no_grad():
            output = alexnet(image)
            labels.extend(softmax(output)[:,1].tolist())

    df = pd.DataFrame({'id': ids,
                       'label': labels}, index=ids)
    df.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    main()
