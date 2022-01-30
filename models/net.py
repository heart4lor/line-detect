import torch
import torch.nn as nn


class basic_conv(nn.Module):
    #  (conv => BN => ReLU)
    def __init__(self, in_ch, out_ch, stride):
        super(basic_conv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), padding=1, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        # print(x.size())
        return x


class Net(nn.Module):
    def __init__(self, out_channel):
        super(Net, self).__init__()  # 256x16x1
        self.layers = nn.ModuleList()
        self.layers.append(basic_conv(1, 32, (1, 2)))  # 128x16x32
        self.layers.append(basic_conv(32, 64, (1, 2)))  # 64x16x64
        self.layers.append(basic_conv(64, 128, (1, 2)))  # 32x16x128
        self.layers.append(basic_conv(128, 64, (1, 1)))  # 32x16x64
        self.layers.append(basic_conv(64, 32, (1, 1)))  # 32x16x32
        self.layers.append(basic_conv(32, 16, (1, 1)))  # 32x16x16
        self.layers.append(basic_conv(16, 8, (1, 1)))  # 32x16x8
        self.layers.append(basic_conv(8, 4, (1, 1)))  # 32x16x4
        self.layers.append(basic_conv(4, 2, (1, 1)))  # 32x16x2
        self.layers.append(nn.AvgPool2d((16, 32)))  # global pool

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.mean(x, (2, 3))
        # x = torch.sigmoid(x)
        return x
