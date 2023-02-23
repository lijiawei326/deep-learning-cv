from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root, train: bool, transform=None):
        self.root = root
        self.transform = transform
        if train:
            self.file_path = 'train.txt'
        else:
            self.file_path = 'val.txt'

        with open(self.file_path, 'r') as f:
            self.file_list = [x.strip() for x in f.readlines()]

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.root, 'train', file_name)

        assert os.path.exists(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if 'cat' in file_name:
            label = 0
        elif 'dog' in file_name:
            label = 1
        else:
            raise ValueError

        return img, label

    def __len__(self):
        return len(self.file_list)


class Test_Dataset(Dataset):
    def __init__(self, root, transform=None):
        assert os.path.exists(root)
        self.root = root
        self.file_list = sorted(os.listdir(root), key=lambda x: int(str(x).split('.')[0]))
        self.transform = transform

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root, file_name)
        id = int(file_name.split('.')[0])

        assert file_path
        img = Image.open(file_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, id

    def __len__(self):
        return len(self.file_list)


# transform = transforms.Compose([transforms.ToTensor()])
# dataset = MyDataset('./data',train=True,transform=transform)
# loader = DataLoader(dataset,batch_size=1)
# for data in loader:
#     img,label = data
#     print(img.shape)
#     print(label)
#     break

# test_dataset = Test_Dataset(root='./data/test', transform=transform)
# loader = DataLoader(test_dataset, batch_size=1)
# for img, id in loader:
#     print(id)
#     break