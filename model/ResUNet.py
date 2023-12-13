import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,in_channels, mid_channels, out_channels,stride=[1,1,1],padding=[0,1,0],first=False) -> None:
        super(ResBlock,self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=1,stride=stride[0],padding=padding[0],bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(mid_channels,mid_channels,kernel_size=3,stride=stride[1],padding=padding[1],bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(mid_channels,out_channels,kernel_size=1,stride=stride[2],padding=padding[2],bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out1 = self.shortcut(x)
        out += out1
        out = F.relu(out)
        return out



class ResDoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResDoubleConv, self).__init__()
        self.res1 = ResBlock(in_channel, out_channel//2, out_channel)
        self.res2 = ResBlock(out_channel, out_channel//2, out_channel)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class InConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InConv, self).__init__()
        self.conv = ResDoubleConv(in_channel, out_channel)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(in_channel, out_channel)
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
        self.conv = DoubleConv(in_channel, out_channel)

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


class ResUNet(nn.Module):
    def __init__(self, n_channel, n_classes=3):
        super(ResUNet, self).__init__()
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
    # img = torch.randn(8, 1, 224, 224).cuda()
    # model = ResUNet(1, 1).cuda()
    # out = model(img)
    # print(out.size())
    
    
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(summary(ResUNet(1, 1).to(device), input_size=(1, 256, 256), batch_size=-1))
    # from torchstat import stat
    # model = UNet(1, 1)
    # stat(model, (1, 224, 224))
    
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameters: %.2fM" % (total/1e6))
