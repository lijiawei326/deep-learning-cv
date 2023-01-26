import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import VGG
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training with {device}!')

    transform = {'train': transforms.Compose([  # transforms.Resize((224,224)),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4883, 0.4551, 0.4170],
                             [0.2294, 0.2250, 0.2252])]),
        'val': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.4883, 0.4551, 0.4170], [0.2294, 0.2250, 0.2252])])}

    train_dataset = MyDataset(root='./data', train=True, transform=transform['train'])
    val_dataset = MyDataset(root='./data', train=False, transform=transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=48)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=True, num_workers=48)

    net = VGG(model_name='vgg16', init_weights=True)
    # 预训练模型
    if pretrain:
        weights_path = './vgg16-397923af.pth'
        pretrained_dict = torch.load(weights_path, map_location='cpu')
        model_dict = net.state_dict()
        # pretrained_dict = {k: v for k, v in weights_dict.items() if k in model_dict.keys()}
        del pretrained_dict['classifier.6.weight']
        del pretrained_dict['classifier.6.bias']
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=3e-2, weight_decay=1e-4)

    tb_writer = SummaryWriter(log_dir='./logs')
    save_path = './vgg16.pth'
    epochs = 100
    steps = len(train_loader)
    best_acc = 0.0
    best_val_loss = np.Inf
    for epoch in range(epochs):
        start_time = time.time()
        # train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            img, label = data
            img, label = img.to(device), label.to(device)

            # 梯度清零，前向传播
            optimizer.zero_grad()
            output = net(img)

            # 计算损失，反向传播
            loss = loss_function(output, label)
            loss.backward()

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

        # val
        # acc = 0.0
        net.eval()

        with torch.no_grad():
            val_loss = 0.0
            for data in val_loader:
                img, label = data
                img, label = img.to(device), label.to(device)

                output = net(img)
                val_loss += loss_function(output, label).item()
                # predict_y = torch.max(output, dim=1)[1]
                # acc += torch.eq(predict_y, label).sum().item()

            # acc = acc / len(val_dataset)

        end_time = time.time()
        running_time = end_time - start_time

        # tb_writer.add_scalar('accuracy', acc, epoch + 1)
        tb_writer.add_scalars('loss', {'train': val_loss / len(val_loader),
                                       'val': running_loss / steps}, epoch + 1)
        print(f'epoch : {epoch + 1}:\n'
              f'       loss : {running_loss / steps}\n'
              # f'       acc : {acc}\n'
              f'   val_loss : {val_loss / len(val_loader)}\n'
              f'       time : {int(running_time // 60)}m {int(running_time % 60)}s')

        # if best_acc <= acc:
        #     torch.save(net.state_dict(), save_path)
        #     best_acc = acc
        if best_val_loss >= val_loss / len(val_loader):
            torch.save(net.state_dict(), save_path)
            best_val_loss = val_loss / len(val_loader)
    tb_writer.close()
    print('Training Finished!')


if __name__ == '__main__':
    pretrain = True
    main()
