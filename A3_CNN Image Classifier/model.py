import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from pytorch_lightning import LightningModule


# 初始化卷积层和全连接层的权重与偏置
def weight_init(module: nn.Module):
    classname = module.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(module.bias)


class SimpleCIFAR10Classifier(LightningModule):
    def __init__(self, config: dict):
        super(SimpleCIFAR10Classifier, self).__init__()

        self.linear_size_1 = 128
        self.linear_size_2 = 32
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.accuracy = Accuracy(task='multiclass', num_classes=10, top_k=1)

        self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm_1 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 6, 3)
        self.conv4 = nn.Conv2d(6, 16, 3)
        self.batch_norm_2 = nn.BatchNorm2d(16)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(16 * 5 * 5, self.linear_size_1)
        self.fc2 = nn.Linear(self.linear_size_1, self.linear_size_2)
        self.fc3 = nn.Linear(self.linear_size_2, 10)
        self.eval_loss = []
        self.eval_accuracy = []

        self.apply(weight_init)

    @staticmethod
    def cross_entropy_loss(logits, labels):
        return F.nll_loss(logits, labels)

    def forward(self, x):
        # 在卷积部分加入批归一化，全连接部分加入dropout；卷积层如此安排的目的是用2 * 2个(3, 3)卷积核模拟2个(5, 5)卷积核
        x = F.relu(self.conv1(x))
        x = self.pool(self.batch_norm_1(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = self.pool(self.batch_norm_2(F.relu(self.conv4(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("logs/train_loss", loss)
        self.log("logs/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("logs/val_loss", avg_loss)
        self.log("logs/val_accuracy", avg_acc)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
