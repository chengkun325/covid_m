import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s

class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x


class VGG16BN(nn.Module):
    def __init__(self, input_features=False):
        super(VGG16BN, self).__init__()
        self.conv1_1 = ConvBNReLU(1, 64, frozen=True)
        self.conv1_2 = ConvBNReLU(64, 64, frozen=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvBNReLU(64, 128, frozen=True)
        self.conv2_2 = ConvBNReLU(128, 128, frozen=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvBNReLU(128, 256, frozen=True)
        self.conv3_2 = ConvBNReLU(256, 256, frozen=True)
        self.conv3_3 = ConvBNReLU(256, 256, frozen=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ConvBNReLU(256, 512, frozen=True)
        self.conv4_2 = ConvBNReLU(512, 512, frozen=True)
        self.conv4_3 = ConvBNReLU(512, 512, frozen=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = ConvBNReLU(512, 512, frozen=True)
        self.conv5_2 = ConvBNReLU(512, 512, frozen=True)
        self.conv5_3 = ConvBNReLU(512, 512, frozen=True)

        self.input_features = input_features
        if input_features:
            print("vgg backbone input features conv_1_2!!!")

    def forward(self, input):
        if not self.input_features:
            conv1_1 = self.conv1_1(input)
            conv1_2 = self.conv1_2(conv1_1)
            pool1 = self.pool1(conv1_2)
        else:
            conv1_2 = input
            pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool3 = self.pool3(conv3_3)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        pool4 = self.pool4(conv4_3)

        conv5_1 = self.conv5_1(conv4_3)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)

        return conv1_2, conv2_2, conv3_3, conv4_3, conv5_3

    def gen_feats(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)

        return conv1_2


def vgg16(pretrained=None, input_features=False):
    model = VGG16BN(input_features=input_features)
    return model

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, stride,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes,stride,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes,stride,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)

        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class FCN(nn.Module):
    def __init__(self, pretrained=None):
        super(FCN, self).__init__()
        self.backbone = resnet18(pretrained)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.cls = nn.Conv2d(512, 1, 1, stride=1, padding=0)

    def forward(self, input):
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)
        saliency_maps = F.interpolate(self.cls(conv5), input.shape[2:], mode='bilinear', align_corners=False)

        return torch.sigmoid(saliency_maps)


class JCS(nn.Module):
    def __init__(self,n_channel, n_classes=3, pretrained=None, use_carafe=True,
                 enc_channels=[64, 128, 256, 512, 512, 512],
                 dec_channels=[64, 128, 256, 512, 512, 512], input_features=False):
        super(JCS, self).__init__()
        self.vgg16 = vgg16(pretrained, input_features=input_features)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gpd = GPD(enc_channels[-1], expansion=4)
        self.gpd1 = GPD(enc_channels[-1], expansion=4, dilation=[1, 2, 3, 4])

        self.fpn1 = FuseModule(enc_channels[0])
        self.fpn2 = FuseModule(enc_channels[1])
        self.fpn3 = FuseModule(enc_channels[2])
        self.fpn4 = FuseModule(enc_channels[3])
        self.fpn5 = FuseModule(enc_channels[4])
        self.fpn5 = FuseModule(enc_channels[5])


        self.channel3 = nn.Conv2d(512,256,kernel_size=1,padding=0)
        self.channel2 = nn.Conv2d(256,128,kernel_size=1,padding=0)
        self.channel1 = nn.Conv2d(128,64,kernel_size=1,padding=0)
        self.pool = nn.MaxPool2d(2,2,0)

        self.input_features = input_features

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = nn.Conv2d(64*n_channel,n_classes,kernel_size=1,padding=0)


    def forward(self, input):
        conv1, conv2, conv3, conv4, conv5 = self.vgg16(input) #224,112,56,28,28,14
        conv5 = self.gpd(conv5)
        conv6 = self.pool(conv5)
        conv6 = self.gpd1(conv6)

        conv6 = self.up(conv6)
        up1 = self.fpn5(conv6,conv5)
        up2 = self.up(self.fpn4(up1,conv4))
        up2 = self.channel3(up2)
        up3 = self.up(self.fpn3(up2,conv3))
        up3 = self.channel2(up3)
        up4 = self.up(self.fpn2(up3,conv2))
        up4 = self.channel1(up4)
        up5 = self.fpn1(up4,conv1)
        out = self.out(up5)

        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // reduction, bias=True)
        self.linear2 = nn.Linear(in_channels // reduction, in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, H, W = x.shape
        embedding = x.mean(dim=2).mean(dim=2)
        fc1 = self.act(self.linear1(embedding))
        fc2 = torch.sigmoid(self.linear2(fc1))
        return x * fc2.view(N, C, 1, 1)


class GPD(nn.Module):
    def __init__(self, in_channels, expansion=4, dilation=[1, 3, 6, 9]):
        super(GPD, self).__init__()
        self.expansion = expansion
        self.expand_conv = ConvBNReLU(in_channels, in_channels * expansion // 2, ksize=1, pad=0, use_bn=False)
        self.reduce_conv = ConvBNReLU(in_channels * expansion // 2, in_channels, ksize=1, pad=0, use_bn=False)
        # self.bn1 = nn.BatchNorm2d(in_channels*expansion//2)
        # self.bn2 = nn.BatchNorm2d(in_channels)
        self.end_conv = ConvBNReLU(in_channels, in_channels, use_relu=True, use_bn=False)
        self.se_block = SEBlock(in_channels * expansion // 2)
        self.dilation_convs = nn.ModuleList()
        for i in dilation:
            self.dilation_convs.append(
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=i, dilation=i))
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.expand_conv(x)
        y = torch.split(y, x.shape[1] // 2, dim=1)
        res = []
        for idx, dilation_conv in enumerate(self.dilation_convs):
            res.append(dilation_conv(y[idx]))
        res = torch.cat(res, dim=1)
        # res = self.bn1(res)
        res = self.act1(res)
        res = self.se_block(res)
        res = self.reduce_conv(res)
        res = self.end_conv(res)
        return res


class FuseModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.se1 = SEBlock(in_channels * 2, in_channels *2)
        self.conv1 = ConvBNReLU(in_channels * 2 , in_channels, use_bn=False)
        self.se2 = SEBlock(in_channels * 2 , in_channels * 2 )
        self.reduce_conv = ConvBNReLU(in_channels * 2 , in_channels, ksize=1, pad=0, use_bn=False)
        self.conv2 = ConvBNReLU(in_channels, in_channels, use_bn=False)

    def forward(self, low, high):
        x = self.se1(torch.cat([low, high], dim=1))
        x = self.conv1(x)
        x = self.se2(torch.cat([x, high], dim=1))
        x = self.reduce_conv(x)
        x = self.conv2(x)
        return x



if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(summary(JCS(1, 1).to(device), input_size=(1,224,224), batch_size=-1))
