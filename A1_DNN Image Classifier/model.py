from torch import nn


class my_DNN(nn.Module):
    def __init__(self):
        super(my_DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=3072, out_features=1500, bias=True),
            nn.LayerNorm(normalized_shape=1500, elementwise_affine=True),
            nn.LeakyReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=1500, out_features=375, bias=True),
            nn.LayerNorm(normalized_shape=375, elementwise_affine=True),
            nn.LeakyReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(in_features=375, out_features=100, bias=True),
            nn.LayerNorm(normalized_shape=100, elementwise_affine=True),
            nn.LeakyReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(in_features=100, out_features=10),
        )

        # 用apply方法递归地初始化参数
        self.apply(weight_init)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


# 调用kaiming初始化方法对线性层权重进行初始化
def weight_init(module: nn.Module):
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(module.weight)
        nn.init.constant_(module.bias, 0.0)
