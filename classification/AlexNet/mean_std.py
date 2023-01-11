import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torchvision.transforms as transforms


class mydataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.file_list = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.file_list[index])

        assert os.path.exists(image_path),f'{image_path} does not exist!'
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image


transform = transforms.Compose([transforms.ToTensor()])
dataset = mydataset(root='./data/train', transform=transform)
train_loader = DataLoader(dataset, batch_size=1)
mean = torch.zeros(3)
std = torch.zeros(3)
for image in train_loader:
    image = image.squeeze()
    for d in range(3):
        mean[d] += image[d, :, :].mean()
        std[d] += image[d, :, :].std()

mean.div_(len(train_loader))
std.div_(len(train_loader))

print(mean, std)
#tensor([0.4883, 0.4551, 0.4170]) tensor([0.2294, 0.2250, 0.2252])