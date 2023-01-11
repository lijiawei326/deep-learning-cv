import os.path
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root, train: bool = True, transform=None):
        self.root = root
        self.transform = transform
        if train:
            file_name = 'train.txt'
        else:
            file_name = 'val.txt'

        assert os.path.exists(file_name),f'{file_name} not exist!'
        with open(file_name, 'r') as f:
            self.file_list = [x.strip() for x in f.readlines()]

    def __getitem__(self, index):
        file_path = self.file_list[index]
        image_path = os.path.join(self.root,'train',file_path)

        assert os.path.exists(image_path),f'{image_path} not exist!'
        image = Image.open(os.path.join(self.root,'train',file_path)).convert('RGB')
        if 'cat' in str(file_path):
            label = 0
        elif 'dog' in str(file_path):
            label = 1
        else:
            raise ValueError

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.file_list)


class test_set(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        if transform is not None:
            self.transform = transform

        self.file_list = sorted(os.listdir(root), key=lambda x: int(str(x).split('.')[0]))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.file_list[index])
        id = int(self.file_list[index].split('.')[0])

        assert os.path.exists(path), f'{path} does not exist!'
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, id