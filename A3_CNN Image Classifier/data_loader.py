from filelock import FileLock
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from pytorch_lightning import LightningDataModule


class Cifar10DataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 2,
                 random_state: int = 10, split_file: str = 'fold_split.pkl'):
        """
        初始化基本参数
        :param data_dir: 数据集加载路径
        :param batch_size:
        :param num_workers: DataLoader中使用的num_workers参数
        :param random_state: 划分数据集所用的随机数种子
        :param split_file: 存储划分结果的文件
        """
        super(Cifar10DataModule, self).__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.random_state = random_state
        self.split_file = split_file

        self.fold_idx = 0

        # 对数据集的预处理和增强
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(32, padding=2),
            v2.RandomHorizontalFlip(),
            # 使用计算得到的训练集RGB均值/方差归一化至[0.0, 1.0]的值
            v2.Normalize((0.49139968, 0.48215841, 0.44653091),
                         (0.24703223, 0.24348513, 0.26158784)),
        ])

    def setup(self, stage: str):
        """
        加载数据集
        """
        with FileLock(f"{self.data_dir}.lock"):
            cifar_10 = CIFAR10(self.data_dir, train=True, transform=self.transform, download=True)
            self.train_dataset, self.val_dataset = random_split(cifar_10, [45000, 5000])
            self.test_dataset = CIFAR10(self.data_dir, train=False, transform=self.transform, download=True)

    # 分别构建训练集、验证集、测试集的DataLoader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
