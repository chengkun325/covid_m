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


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
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

        self.conv = SingleConv(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
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


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True))
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(True))
        self.add_module('conv3', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                           stride=1, padding=1, bias=True))
        self.add_module('norm2', nn.BatchNorm2d(growth_rate))
        self.add_module('relu2', nn.ReLU(True))
        self.add_module('drop', nn.Dropout2d(0.5))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x


class FDUNet(nn.Module):
    def __init__(self, n_channel, n_classes=1, bilinear=False):
        super(FDUNet, self).__init__()
        self.n_channels = n_channel
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = SingleConv(n_channel, 32)
        self.dense_block_down_1 = DenseBlock(32, 8)
        self.down1 = Down()
        self.dense_block_down_2 = DenseBlock(64, 16)
        self.down2 = Down()
        self.dense_block_down_3 = DenseBlock(128, 32)
        self.down3 = Down()
        self.dense_block_down_4 = DenseBlock(256, 64)
        self.down4 = Down()
        self.dense_block_down_5 = DenseBlock(512, 128)
        self.up1 = Up(1024, 256, bilinear)
        self.dense_block_up_1 = DenseBlock(256, 64)
        self.up2 = Up(512, 128, bilinear)
        self.dense_block_up_2 = DenseBlock(128, 32)
        self.up3 = Up(256, 64, bilinear)
        self.dense_block_up_3 = DenseBlock(64, 16)
        self.up4 = Up(128, 32, bilinear)
        self.dense_block_up_4 = DenseBlock(32, 8)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)
        x1 = self.dense_block_down_1(x1)
        x2 = self.down1(x1)
        x2 = self.dense_block_down_2(x2)
        x3 = self.down2(x2)
        x3 = self.dense_block_down_3(x3)
        x4 = self.down3(x3)
        x4 = self.dense_block_down_4(x4)
        x5 = self.down4(x4)
        x5 = self.dense_block_down_5(x5)
        x_tmp = self.up1(x5, x4)
        x_tmp = self.dense_block_up_1(x_tmp)
        x_tmp = self.up2(x_tmp, x3)
        x_tmp = self.dense_block_up_2(x_tmp)
        x_tmp = self.up3(x_tmp, x2)
        x_tmp = self.dense_block_up_3(x_tmp)
        x_tmp = self.up4(x_tmp, x1)
        x_tmp = self.dense_block_up_4(x_tmp)
        logits = self.outc(x_tmp)
        return logits + x


if __name__ == "__main__":
    print(FDUNet(1, 1))
    # from torchsummary import summary
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(summary(FDUNet(1, 1).to(device), input_size=(1, 224, 224), batch_size=-1))
    # t = torch.Tensor([[[[1, 3], [1, 5]]] * 128] * 2)
    # t1 = torch.Tensor([[[[1, 2, 9, 3], [1, 7, 4, 5], [1, 2, 9, 3], [1, 7, 4, 5]]] * 32] * 2)
    # print(t.size())
    # print(t1.size())
    # model = DenseBlock(32, 8)
    # # model = GateAttModule(128, 64)
    # # model = DecoderAttModule(128)
    # g = model.forward(t1)
    # print(g.size())
