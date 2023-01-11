import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import AlexNet
from my_dataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


def main():
    # 定义device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    transfrom = {
        'train': transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.4883, 0.4551, 0.4170], [0.2294, 0.2250, 0.2252])]),
        'val': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.4883, 0.4551, 0.4170], [0.2294, 0.2250, 0.2252])])
    }

    train_dataset = MyDataset(root='./data', train=True, transform=transfrom['train'])
    val_dataset = MyDataset(root='./data', train=False, transform=transfrom['val'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True,num_workers=4)

    # 定义模型
    alexnet = AlexNet()
    alexnet.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=alexnet.parameters(), lr=1e-3)

    epochs = 50
    steps = len(train_loader)
    save_path = './alexnet.pth'
    tb_writer = SummaryWriter(log_dir='./logs')
    best_acc = 0.0
    # 开始训练
    for epoch in range(epochs):
        running_loss = 0.0
        # train
        alexnet.train()
        for step, data in enumerate(train_loader):
            images, label = data
            images = images.to(device)
            label = label.to(device)
            # 梯度清零，前向传播
            optimizer.zero_grad()
            output = alexnet(images)

            # 计算损失，后向传播
            loss = loss_function(output, label)
            loss.backward()

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

        # validate
        alexnet.eval()
        acc = 0.0
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                images, label = data
                images = images.to(device)
                label = label.to(device)

                output = alexnet(images)
                predict_y = torch.max(output, dim=1)[1]
                acc += torch.eq(label, predict_y).sum().item()

            acc = acc / len(val_dataset)

        # 记录训练过程
        tb_writer.add_scalar('loss', running_loss / steps, epoch)
        tb_writer.add_scalar('accuracy', acc, epoch)
        print(f'epoch : {epoch + 1}, loss : {running_loss / steps}, accuracy : {acc}')

        # 保存最优模型参数
        if acc >= best_acc:
            best_acc = acc
            torch.save(alexnet.state_dict(), save_path)

    tb_writer.close()
    print('Training Finished!')


if __name__ == '__main__':
    main()
