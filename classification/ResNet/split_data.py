import os
import random


def main():
    data_path = './data/train'
    dir = os.listdir(data_path)

    cat_dir = []
    dog_dir = []
    for file in dir:
        if 'cat' in file:
            cat_dir.append(file)
        elif 'dog' in file:
            dog_dir.append(file)
        else:
            raise ValueError

    val_rate = 0.2
    cat_val_index = random.sample(range(len(cat_dir)), k=int(len(cat_dir) * val_rate))
    dog_val_index = random.sample(range(len(dog_dir)), k=int(len(dog_dir) * val_rate))

    train_files = []
    val_files = []
    for index, file_name in enumerate(cat_dir):
        if index in cat_val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    for index, file_name in enumerate(dog_dir):
        if index in dog_val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    # 随即化数据分布
    random.shuffle(train_files)
    random.shuffle(val_files)
    try:
        train_f = open('./train.txt', 'w+')
        val_f = open('./val.txt', 'w+')
        train_f.write('\n'.join(train_files))
        val_f.write('\n'.join(val_files))
        train_f.close()
        val_f.close()
    except FileExistsError as e:
        print(e)


if __name__ == '__main__':
    main()
