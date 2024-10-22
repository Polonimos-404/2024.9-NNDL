import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

# 数据集存放路径
DATASET_PATH = 'D:\dev\datasets\CV'

# 原训练集和测试集大小
TRAINSET_SIZE = 50000
TESTSET_SIZE = 10000


# 对数据集的预处理和增强
DATA_TRANSFORM_AUGMENT = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomCrop(32, padding=2),
    v2.RandomHorizontalFlip(),
    # 使用计算得到的训练集RGB均值/方差归一化至[0.0, 1.0]的值
    v2.Normalize((0.49139968, 0.48215841, 0.44653091),
                 (0.24703223, 0.24348513, 0.26158784)),
])


# 加载数据集
def load_data(train: bool, batch_size: int = 64, val_scale: float = 0.1):
    """

    :param train: 加载训练集或测试集
    :param batch_size:
    :param val_scale: 验证集占原数据集比例；如果train == True且val_scale == 0.0，则加载整个训练集
    :return:
    """
    # 加载数据集
    cifar_10 = CIFAR10(
        root=DATASET_PATH,
        train=train,
        transform=DATA_TRANSFORM_AUGMENT,
    )

    # 包装为DataLoader
    def wrap_loader(dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

    if train and val_scale > 0.0:
        # 将数据集按比例划分为训练集和验证集，同时各类别数据比例不变
        train_indices, val_indices = train_test_split(
            [i for i in range(TRAINSET_SIZE)],
            stratify=cifar_10.targets,
            test_size=val_scale,
            random_state=42
        )
        train_set = Subset(cifar_10, train_indices)
        val_set = Subset(cifar_10, val_indices)
        return wrap_loader(train_set), wrap_loader(val_set)
    else:
        return wrap_loader(cifar_10)
