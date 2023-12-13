from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import torch

class HSBlock(nn.Module):
    '''
    替代3x3卷积
    '''
    def __init__(self, in_ch, s=8):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()

        in_ch_range=torch.Tensor(in_ch)
        in_ch_list = list(in_ch_range.chunk(chunks=self.s, dim=0))

        self.module_list.append(nn.Sequential())
        channel_nums = []
        for i in range(1,len(in_ch_list)):
            if i == 1:
                channels = len(in_ch_list[i])
            else:
                random_tensor = torch.Tensor(channel_nums[i-2])
                _, pre_ch = random_tensor.chunk(chunks=2, dim=0)
                channels= len(pre_ch)+len(in_ch_list[i])
            channel_nums.append(channels)
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]



class HSBResConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HSBResConv, self).__init__()
        self.conv = nn.Sequential(
            HSBlock(in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1) if in_channel != out_channel else nn.Sequential()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        identity = self.shortcut(identity)
        x = x+identity
        return x
    
class InConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InConv, self).__init__()
        self.conv = HSBResConv(in_channel, out_channel)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            HSBResConv(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.mp_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channel, out_channel, biliner=True):
        super(Up, self).__init__()
        if biliner:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channel // 2, in_channel // 2, 2, stride=2)
        self.conv = HSBResConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x1.size()[2] - x2.size()[2]  # 得到图像x2与x1的H的差值，56-64=-8
        diffX = x1.size()[3] - x2.size()[3]  # 得到图像x2与x1的W差值，56-64=-8
        x2 = F.pad(x2, (diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class HSBUNet(nn.Module):
    def __init__(self, n_channel, n_classes=3):
        super(HSBUNet, self).__init__()
        self.inc = InConv(n_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.size())
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x




if __name__ == "__main__":
    # from torchsummary import summary

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = HSBUNet(1, 1).cuda()
    # in_img = torch.randn(8, 1, 224, 224).cuda()
    # y = model(in_img)
    # print(y.size())
    from torchstat import stat
    model = HSBUNet(1, 1)
    stat(model, (1, 224, 224))
# end main