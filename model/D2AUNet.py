import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.gam = GateAttModule(in_channels, out_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x2 = self.gam(x1, x2)
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DecoderAttModule(nn.Module):
    def __init__(self, in_channels):
        super(DecoderAttModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv_3_3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv_5_5 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.conv_7_7 = nn.Conv2d(1, 1, kernel_size=7, padding=3)
        self.fc_1 = nn.Linear(in_channels, in_channels)
        self.fc_2 = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        sp_att = self.conv1(x)
        sp_att_3 = self.conv_3_3(sp_att)
        sp_att_5 = self.conv_5_5(sp_att)
        sp_att_7 = self.conv_7_7(sp_att)
        sp_att = sp_att_3 + sp_att_5 + sp_att_7
        sp_att = torch.sigmoid(sp_att)
        ch_att = F.adaptive_max_pool2d(x, (1, 1)).squeeze()
        ch_att = self.fc_1(ch_att)
        ch_att = F.relu(ch_att)
        ch_att = self.fc_2(ch_att).unsqueeze(-1).unsqueeze(-1)
        ch_att = torch.sigmoid(ch_att)

        return sp_att * x * ch_att


class ResAttBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResAttBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.dam = DecoderAttModule(in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = self.relu(x1)
        x1 = self.dam(x1)
        return x + x1


class GateAttModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GateAttModule, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv_2 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.fc_1 = nn.Linear(in_channels, out_channels)
        self.fc_2 = nn.Linear(out_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_3_3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv_5_5 = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.conv_7_7 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        # self.conv_sp_1 = nn.Conv2d()

    def forward(self, g, f):
        g1 = self.conv_1(g)
        g1 = self.up(g1)
        f1 = self.conv_2(f)
        f_g = torch.cat((g1, f1), 1)
        f_g_3 = self.conv_3_3(f_g)
        f_g_5 = self.conv_5_5(f_g)
        f_g_7 = self.conv_7_7(f_g)
        f_g = f_g_3+f_g_5+f_g_7
        f_g = torch.sigmoid(f_g)
        g2 = F.adaptive_max_pool2d(g, (1, 1)).squeeze()
        g2 = self.fc_1(g2)
        g2 = F.relu(g2)
        g2 = self.fc_2(g2).unsqueeze(-1).unsqueeze(-1)
        g2 = torch.sigmoid(g2)

        return g2 * f_g


class D2AUNet(nn.Module):
    def __init__(self, n_channel, n_classes=3, bilinear=False):
        super(D2AUNet, self).__init__()
        self.n_channels = n_channel
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.rab1 = ResAttBlock(512)
        self.up2 = Up(512, 256, bilinear)
        self.rab2 = ResAttBlock(256)
        self.up3 = Up(256, 128, bilinear)
        self.rab3 = ResAttBlock(128)
        self.up4 = Up(128, 64, bilinear)
        self.rab4 = ResAttBlock(64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.rab1(x)
        x = self.up2(x, x3)
        x = self.rab2(x)
        x = self.up3(x, x2)
        x = self.rab3(x)
        x = self.up4(x, x1)
        x = self.rab4(x)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    from torchsummary import summary



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(summary(D2AUNet(1, 1).to(device), input_size=(1,224,224), batch_size=-1))
    # t = torch.Tensor([[[[1, 3], [1, 5]]] * 128] * 2)
    # t1 = torch.Tensor([[[[1, 2, 9, 3], [1, 7, 4, 5], [1, 2, 9, 3], [1, 7, 4, 5]]] * 64] * 2)
    # print(t.size())
    # print(t1.size())
    # # model = GateAttModule(128, 64)
    # model = DecoderAttModule(128)
    # g = model.forward(t)
    # print(g.size())
