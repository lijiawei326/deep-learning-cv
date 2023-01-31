import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import GooLeNet
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Training with {device}!')

    transform = {'train': transforms.Compose([  # transforms.Resize((224,224)),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4883, 0.4551, 0.4170],
                             [0.2294, 0.2250, 0.2252])]),
        'val': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.4883, 0.4551, 0.4170], [0.2294, 0.2250, 0.2252])])}

    train_dataset = MyDataset(root='./data', train=True, transform=transform['train'])
    val_dataset = MyDataset(root='./data', train=False, transform=transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=48)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=48)

    net = GooLeNet(aux_logits=args.aux, init_weights=True)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    tb_writer = SummaryWriter(log_dir=args.logs)
    save_path = args.save_path
    epochs = args.epochs
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

            optimizer.zero_grad()

            if args.aux:
                logits, aux1_logits, aux2_logits = net(img)
                loss0 = loss_function(logits, label)
                loss1 = loss_function(aux1_logits, label)
                loss2 = loss_function(aux2_logits, label)
                loss = loss0 + 0.3*loss1 + 0.3*loss2
            else:
                logits = net(img)
                loss = loss_function(logits, label)
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
        tb_writer.add_scalars('loss', {'train': running_loss / steps,
                                       'val': val_loss / len(val_loader)}, epoch + 1)
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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 文件保存名称
    parser.add_argument('--save-path', default='GoogLeNet.pth', help='weights_path')
    # 训练epoch数
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=3e-2, type=float, help='learning rate')
    # tensorboard logs保存地址
    parser.add_argument('--logs', default='./logs', help='tensorboard logs saving path')
    # 是否使用辅助分类器
    parser.add_argument('--aux', default=False, type=bool, help='use auxiliary classifier or not')

    args = parser.parse_args()
    main(args)
