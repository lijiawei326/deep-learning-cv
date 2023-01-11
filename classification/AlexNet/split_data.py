import os
import random


def main():
    random.seed(0)
    dir = os.listdir('./data/train')
    val_rate = 0.05
    file_num = len(dir)
    val_index = random.sample(range(0,file_num), k=int(file_num*val_rate))

    train_files = []
    val_files =[]

    for index,file_name in enumerate(dir):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open('train.txt','a+')
        val_f = open('val.txt','a+')
        train_f.write('\n'.join(train_files))
        val_f.write('\n'.join(val_files))
        train_f.close()
        val_f.close()
    except FileExistsError as e:
        print(e)


if __name__ == '__main__':
    main()
