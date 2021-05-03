import torch
import torch.nn as nn
import torch.nn.functional as F


# 一个卷积残差块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes,
                               out_channels=planes,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(in_channels=planes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(in_channels=planes,
                               out_channels=self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.expansion * planes)

        self.shortcut = nn.Sequential()

        # 步长不为 1 或者 输入特征不等于输出特征
        if stride != 1 or in_planes != self.expansion * planes:
            # 残差块
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes,
                          out_channels=self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers, backbone of the network
        # planes 是输出特征
        # channel 变化：3 -> 64 -> 64 -> 256
        self.layer1 = self._make_layer(block=block,
                                       planes=64,
                                       num_blocks=num_blocks[0],
                                       stride=1)
        # channel 变化：64*4 -> 128 -> 128 -> 512
        self.layer2 = self._make_layer(block=block,
                                       planes=128,
                                       num_blocks=num_blocks[1],
                                       stride=2)
        # out_channel: 1024
        self.layer3 = self._make_layer(block=block,
                                       planes=256,
                                       num_blocks=num_blocks[2],
                                       stride=2)
        # out_channel 2048
        self.layer4 = self._make_layer(block=block,
                                       planes=512,
                                       num_blocks=num_blocks[3],
                                       stride=2)

        # Top layer
        # layer4 后面接一个1x1, 256 conv，得到金字塔最顶端的feature
        self.toplayer = nn.Conv2d(in_channels=2048,
                                  out_channels=256,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

        # Smooth layers
        # 这个是上面引文中提到的抗 『混叠』 的3x3卷积
        # 由于金字塔上的所有feature共享classifier和regressor
        # 要求它们的channel dimension必须一致
        # 这个用于多路预测
        self.smooth1 = nn.Conv2d(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.smooth2 = nn.Conv2d(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.smooth3 = nn.Conv2d(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        # Lateral layers
        # 为了匹配channel dimension引入的1x1卷积
        # 注意这些backbone之外的extra conv，输出都是256 channel
        self.latlayer1 = nn.Conv2d(in_channels=1024,
                                   out_channels=256,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(in_channels=512,
                                   out_channels=256,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer3 = nn.Conv2d(in_channels=256,
                                   out_channels=256,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    ## FPN的lateral connection部分: upsample以后，element-wise相加
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # 上采样到指定尺寸
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        # P5: 金字塔最顶上的feature 2048 -> 256
        p5 = self.toplayer(c5)
        # P4: 上一层 p5 + 侧边来的 c4
        # 其余同理
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


def FPN101():
    return FPN(Bottleneck, [2, 2, 2, 2])


def test():
    net = FPN101()
    fms = net(torch.randn((1, 3, 600, 900), requires_grad=True))
    for fm in fms:
        print(fm.size())


test()

# ref: https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py