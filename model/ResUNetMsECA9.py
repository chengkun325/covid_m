import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECABlock(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()
 
        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
 
        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
 
        # 激活函数
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)
 
        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return x * v
    
class ECAWeight(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ECAWeight, self).__init__()
 
        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
 
        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
 
        # 激活函数
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)
 
        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return v

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out



class MultiScaleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MultiScaleConv, self).__init__()
        assert in_channel % 4 == 0, "The num of in_channel can not be divided by 4."
        
        # assert out_channel % in_channel == 0, "The num of out_channel can not be divided by the num of in_channel."
        
        self.single_ch = in_channel // 4
        self.expansion = out_channel // in_channel
        
        self.conv3_3 = nn.Conv2d(self.single_ch, self.single_ch, kernel_size=3, stride=1, padding=1)
        
        self.atrous_conv3 = nn.Conv2d(self.single_ch, self.single_ch, 3, 1, padding=3, dilation=3)
        self.atrous_conv5 = nn.Conv2d(self.single_ch, self.single_ch, 3, 1, padding=5, dilation=5)
        self.atrous_conv7 = nn.Conv2d(self.single_ch, self.single_ch, 3, 1, padding=7, dilation=7)
        self.eca = ECABlock(self.single_ch * 7)
        self.initialize_weights()
    
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
        x_split = x.chunk(chunks=4, dim=1)
        x0 = self.conv3_3(x_split[0])
        x1 = self.atrous_conv3(x_split[1])
        x2 = self.atrous_conv5(x_split[2])
        x3 = self.atrous_conv7(x_split[3])
        
        x4 = torch.add(x0, x1)
        x5 = torch.add(x1, x2)
        x6 = torch.add(x2, x3)
        
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6], dim=1)
        x = self.eca(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,in_channels, mid_channels, out_channels,stride=[1,1,1],padding=[0,1,0],first=False) -> None:
        super(ResBlock,self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=1,stride=stride[0],padding=padding[0],bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            MultiScaleConv(mid_channels, mid_channels//4 * 7),
            # nn.Conv2d(mid_channels,mid_channels,kernel_size=3,stride=stride[1],padding=padding[1],bias=False),
            nn.BatchNorm2d(mid_channels // 4 * 7),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(mid_channels//4*7,out_channels,kernel_size=1,stride=stride[2],padding=padding[2],bias=False),
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
    
class FeatureEnchanceModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureEnchanceModule, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.eca_weight = ECAWeight(out_channel)

    def forward(self, x):
        x_conv = self.conv(x)
        x_eca = self.eca_weight(x)
        
        x_s = self.sigmoid(x_conv)
        x_m = torch.mul(x_s, x_conv)
        x_add = torch.add(x_m, x_conv)
        
        return x_add * x_eca


class RConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RConv, self).__init__()
        self.conv = nn.Sequential(
            ResBlock(in_channel, out_channel, out_channel),
            FeatureEnchanceModule(out_channel, out_channel)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class InConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InConv, self).__init__()
        self.conv = RConv(in_channel, out_channel)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            RConv(in_channel, out_channel)
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




class ResUNetMsECA9(nn.Module):
    def __init__(self, n_channel, n_classes=3):
        super(ResUNetMsECA9, self).__init__()
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
    # img = torch.randn(8, 1, 224, 224).cuda()
    # model = ResUNet(1, 1).cuda()
    # out = model(img)
    # print(out.size())
    
    
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(summary(ResUNetMsECA9(1, 1).to(device), input_size=(1, 256, 256), batch_size=-1))
    # from torchstat import stat
    # model = UNet(1, 1)
    # stat(model, (1, 224, 224))
    
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameters: %.2fM" % (total/1e6))
