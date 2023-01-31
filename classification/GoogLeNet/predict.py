from my_dataset import Test_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from model import GooLeNet
import pandas as pd
import datetime


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Predicting with {device}!')

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4883, 0.4551, 0.4170], [0.2294, 0.2250, 0.2252])]
                                   )

    test_dataset = Test_Dataset(root='./data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=48, num_workers=48)

    weights_path = './NiN.pth'
    assert weights_path
    weights_dict = torch.load(weights_path, map_location='cpu')
    net = GooLeNet(init_weights=True)
    net.load_state_dict(weights_dict)
    net.to(device)

    ids = []
    labels = []
    softmax = nn.Softmax(dim=1)
    net.eval()
    with torch.no_grad():
        for img, id in test_loader:
            img = img.to(device)
            output = softmax(net(img))[:, 1]

            ids.extend(id.tolist())
            labels.extend(output.tolist())

    df = pd.DataFrame({'id': ids,
                       'label': labels})
    df.to_csv(f'./submission-{datetime.date.today()}.csv', index=False)


if __name__ == '__main__':
    main()
