from pipeline import pipeline


# SGD调参
def sgd_tune():
    param_lists = {
        'lr': [0.1, 0.01, 0.001],
        'momentum': [0, 0.8, 0.9],
        'weight_decay': [0.0001, 0.00001]
    }
    my_pip = pipeline()
    my_pip.run('SGD', 'SGD tune 1', **param_lists)


# 其他优化器测试
def main():
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-5

    my_pip = pipeline()

    my_pip.run('SGD', 'SGD', "../SGD tune 1/models/11 (0.01, 0.9, 1e-05).pth",
               lr=lr, momentum=momentum, weight_decay=weight_decay)

    my_pip.run('Adagrad', 'Adagrad', lr=[lr], weight_decay=[weight_decay], eps=[1e-8])
    my_pip.run('Adagrad', 'Adagrad', './models/0 (0.01, 1e-05, 1e-08).pth', lr=lr, weight_decay=weight_decay, eps=1e-8)

    my_pip.run('RMSprop', 'RMSprop', lr=[lr], weight_decay=[weight_decay], momentum=[momentum])
    my_pip.run('RMSprop', 'RMSprop', './models/0 (0.01, 1e-05, 0.9).pth', lr=lr, weight_decay=weight_decay, momentum=momentum)

    my_pip.run('Adam', 'Adam', lr=[lr], weight_decay=[weight_decay])
    my_pip.run('Adam', 'Adam', './models/0 (0.01, 1e-05).pth', lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    main()
