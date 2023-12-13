import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
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

class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.conv = nn.Conv2d(in_channel, depth//4, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block6 = nn.Conv2d(in_channel, depth//4, 3, 1, padding=3, dilation=3)
        self.atrous_block12 = nn.Conv2d(in_channel, depth//4, 3, 1, padding=5, dilation=5)
        self.atrous_block18 = nn.Conv2d(in_channel, depth//4, 3, 1, padding=7, dilation=7)
 
        self.shortcut = nn.Conv2d(in_channel, depth, 1, 1)
 
    def forward(self, x):
        shortcut = self.shortcut(x)
        image_features = self.conv(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        x = torch.cat([image_features, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1)
        x += shortcut
        return x

class UNetASPP(nn.Module):
    def __init__(self, n_channel, n_classes=3):
        super(UNetASPP, self).__init__()
        self.inc = InConv(n_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.aspp = ASPP(512, 512)
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
        x5 = self.aspp(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(summary(UNetASPP(1, 1).to(device), input_size=(1, 224, 224), batch_size=-1))
    # from torchstat import stat
    # model = UNet(1, 1)
    # stat(model, (1, 224, 224))
    
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameters: %.2fM" % (total/1e6))
