import os
from copy import deepcopy
from itertools import product
from time import time

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim

from data_loader import load_data, TRAINSET_SIZE, TESTSET_SIZE
from model import my_DNN

OPTIMIZER = {
    'SGD': optim.SGD,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'RMSprop': optim.RMSprop
}


class pipeline:
    def __init__(self, batch_size: int = 64, val_scale: float = 0.1):
        # 训练集，验证集
        self.train_loader, self.val_loader = load_data(train=True, batch_size=batch_size, val_scale=val_scale)
        self.val_size = int(val_scale * TRAINSET_SIZE)
        self.train_size = TRAINSET_SIZE - self.val_size
        # 原训练集
        self.train_full_loader = load_data(train=True, batch_size=batch_size, val_scale=0.0)
        self.train_full_size = TRAINSET_SIZE
        # 测试集
        self.test_loader = load_data(train=False, batch_size=batch_size)
        self.test_size = TESTSET_SIZE

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}.')

        self.model = my_DNN().to(self.device)
        # 维护模型初始参数以用于重置模型
        self.init_state_dict = deepcopy(self.model.state_dict())
        # 固定使用交叉熵损失作为损失函数
        self.criterion = nn.CrossEntropyLoss()

        print(f'PyTorch Version: {torch.__version__}')

    # 计算损失和准确率
    def cal_loss_and_num_correct(self, feature, label):
        output = self.model(feature)
        loss = self.criterion(output, label)
        _, preds = output.max(1)
        num_correct = (preds == label).sum()
        return loss, num_correct

    # noinspection PyUnboundLocalVariable
    # 训练+验证工作流：在划分后的训练集上训练模型，并在验证集上验证，重复固定的轮数
    def train_and_evaluate(self, optimizer, epochs: int = 20):
        print(f'Training and validation begins.\n'
              f'Train : Val = {self.train_size}\n'
              f'-Epoch set at {epochs}.')

        # 记录每一epoch训练和验证的损失函数和准确率
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        start_time = time()
        for epoch in range(epochs):
            print(f"_____Epoch {epoch + 1}_____")

            # 训练
            train_loss = 0.0
            train_correct = 0
            self.model.train()
            # MBGD
            for feature, label in self.train_loader:
                feature = feature.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.long)

                optimizer.zero_grad()
                loss, num_correct = self.cal_loss_and_num_correct(feature, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += num_correct

            train_acc = train_correct / self.train_size * 100
            print(f'Training: Loss: {train_loss}    Acc: {train_correct} / {self.train_size}  {train_acc:.2f}%')
            # 保存该轮训练的损失和准确率
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc.cpu().tolist())

            # 验证
            val_loss = 0.0
            val_correct = 0
            self.model.eval()
            with torch.no_grad():
                for feature, label in self.val_loader:
                    feature = feature.to(device=self.device, dtype=torch.float32)
                    label = label.to(device=self.device, dtype=torch.long)

                    loss, num_correct = self.cal_loss_and_num_correct(feature, label)
                    val_loss += loss.item()
                    val_correct += num_correct

            val_acc = val_correct / self.val_size * 100
            print(f'Validation: Loss: {val_loss}    Acc: {val_correct} / {self.val_size}  {val_acc:.2f}%')
            # 保存该轮验证的损失和准确率
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc.cpu().tolist())

            print(f'Time elapsed: {time() - start_time:.4f}s\n'
                  f'__________________\n')
        print(f'INFO: finished in {time() - start_time:.4f}s.\n\n')
        return train_loss_list, train_acc_list, val_loss_list, val_acc_list, epoch + 1

    # 在划分前的CIFAR-10的训练集上训练，并在测试集上测试
    def train_and_test(self, optimizer, epochs: int = 10):
        # 训练
        print(f'Training on whole trainset begins.\n'
              f'-Epoch set at {epochs}.')
        train_loss_list = []
        train_acc_list = []
        start_time = time()
        for epoch in range(epochs):
            print(f"_____Epoch {epoch + 1}_____")

            train_loss = 0.0
            train_correct = 0
            self.model.train()

            for feature, label in self.train_full_loader:
                feature = feature.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.long)

                optimizer.zero_grad()
                loss, num_correct = self.cal_loss_and_num_correct(feature, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += num_correct

            train_acc = train_correct / self.train_full_size * 100
            print(f'Training: Loss: {train_loss}    Acc: {train_correct} / {self.train_full_size}  {train_acc:.2f}%')
            # 保存该轮训练的损失和准确率
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc.cpu().tolist())

            print(f'Time elapsed: {time() - start_time:.4f}s\n'
                  f'__________________\n')
        print(f'INFO: training finished in {time() - start_time:.4f}s.\n\n')

        # 测试
        print('Test begins.')
        test_correct = 0
        predss = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for feature, label in self.test_loader:
                feature = feature.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.long)

                output = self.model(feature)
                _, preds = output.max(1)
                num_correct = (preds == label).sum()
                test_correct += num_correct
                predss += preds.cpu().tolist()
                labels += label.cpu().tolist()
        # 计算准确率和混淆矩阵
        test_acc = test_correct / self.test_size * 100
        conf_mtx = confusion_matrix(labels, predss).tolist()
        print(f'Test Accuracy: {test_acc}')
        print('Confusion Matrix: ')
        for row in conf_mtx:
            print(' '.join([str(i) for i in row]))

        return train_loss_list, train_acc_list, test_acc, conf_mtx, epoch + 1

    # 绘制和保存评价指标变化曲线
    @staticmethod
    def draw_line_chart(param_text: str,
                        train_loss_list, train_acc_list, val_loss_list, val_acc_list,
                        epoch_limit: int = 20):
        """

        :param param_text: 超参字符串，用于图标题
        :param train_loss_list:
        :param train_acc_list:
        :param val_loss_list:
        :param val_acc_list:
        :param epoch_limit:
        :return:
        """
        epoch = [i for i in range(len(train_loss_list))]

        plt.figure(dpi=300, figsize=(7, 4.8))
        # 绘制损失变化曲线
        plt.subplot(2, 1, 1)
        plt.plot(epoch, train_loss_list, label='train', color='blue', marker='o')
        plt.plot(epoch, val_loss_list, label='validate', color='orange', marker='o')
        # 设置坐标轴范围
        plt.xlim((0, epoch_limit))
        # plt.ylim((0, 105))
        # 设置坐标轴刻度
        plt.xticks([i for i in range(0, epoch_limit + 1, 5)])
        # plt.yticks([i for i in range(0, 101, 10)])
        # 设置坐标轴名称
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # 设置图标题
        plt.title('Train Loss vs Val Loss\n' + param_text)

        # 绘制准确率变化曲线
        plt.subplot(2, 1, 2)
        plt.plot(epoch, train_acc_list, label='train', color='blue', marker='o')
        plt.plot(epoch, val_acc_list, label='validate', color='orange', marker='o')
        plt.xlim((0, epoch_limit))
        plt.ylim((0, 105))
        plt.xticks([i for i in range(0, epoch_limit + 1, 5)])
        plt.yticks([i for i in range(0, 101, 10)])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train Acc vs Val Acc\n' + param_text)
        # 调整子图间距
        plt.subplots_adjust(hspace=0.6)
        plt.legend()
        # 保存
        fig = plt.gcf()
        fig.savefig('./results/' + param_text[8:] + '.png')

    def run(self, optim_type_name: str = 'SGD', save_folder_name: str = None, load_path: str = None, **param_lists):
        """

        :param optim_type_name: 优化器类型，将会从OPTIMIZER中按键值对匹配
        :param save_folder_name: 存储工作结果（模型、曲线图和日志）的文件夹
        :param load_path: 训练+测试分支下加载模型state_dict的路径
        :param param_lists: 用于初始化优化器的超参数
        :return:
        """
        optim_type = OPTIMIZER[optim_type_name]
        # 训练+验证分支
        if isinstance(param_lists['lr'], list):
            # 如果文件夹名不存在则创建文件夹和子文件夹，否则返回错误
            if not os.path.exists(save_folder_name):
                os.mkdir('./' + save_folder_name)
                os.chdir('./' + save_folder_name)
            else:
                print('WARN: Folder name already exists.')
                return
            os.mkdir('./models')
            os.mkdir('./results')

            f = open('log.txt', 'w')
            # 逐个参数组合执行
            for i, values in enumerate(product(*param_lists.values())):
                name_values = dict(zip(param_lists.keys(), values))
                # 维护一个列出所有超参的字符串
                param_text = str('Params: ' + ', '.join(f'{key} = {value}' for key, value in name_values.items()))
                print(param_text)

                optimizer = optim_type(self.model.parameters(), **name_values)
                train_loss_list, train_acc_list, val_loss_list, val_acc_list, epoch \
                    = self.train_and_evaluate(optimizer)

                # 绘制和保存曲线图，保存模型state_dict
                self.draw_line_chart(param_text, train_loss_list, train_acc_list, val_loss_list, val_acc_list)
                torch.save(self.model.state_dict(), f'./models/{i} {values}.pth')
                # 重置模型
                self.model.load_state_dict(self.init_state_dict)
                # 写入日志
                f.write(f'{i}\n')
                f.write(param_text + '\n')
                f.write(f'epoch: {epoch}\n')
                f.write(f'train_loss_list: {train_loss_list}\n')
                f.write(f'train_acc_list: {train_acc_list}\n')
                f.write(f'val_loss_list: {val_loss_list}\n')
                f.write(f'val_acc_list: {val_acc_list}\n\n')
                f.flush()
            f.close()
        # 训练+测试分支
        else:
            os.chdir(save_folder_name)
            # 从路径中加载state_dict
            state_dict = torch.load(load_path, weights_only=True)
            self.model.load_state_dict(state_dict)
            optimizer = optim_type(self.model.parameters(), **param_lists)
            train_loss_list, train_acc_list, test_acc, conf_mtx, epoch = self.train_and_test(optimizer)
            torch.save(self.model.state_dict(), './model.pth')
            # 写入日志
            with open('result.txt', 'w') as f:
                f.write('Train: \n')
                f.write(f'epoch: {epoch}\n')
                f.write(f'train_loss_list: {train_loss_list}\n')
                f.write(f'train_acc_list: {train_acc_list}\n\n')
                f.write('Test: \n')
                f.write(f'Acc: {test_acc}\n')
                f.write(f'Confusion Matrix: \n{conf_mtx}')

        os.chdir('..')
        self.model.load_state_dict(self.init_state_dict)
