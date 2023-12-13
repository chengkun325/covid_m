import torch
from torch import nn
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, middle_channels)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = conv3x3(middle_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out


class ResBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, middle_channels)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(middle_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

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


class SAmodule(nn.Module):
    def __init__(self, F_1, F_2, F_3):
        super(SAmodule, self).__init__()
        self.W_x = nn.Sequential(
            conv1x1(F_1, F_3),
            nn.BatchNorm2d(F_3)
        )
        self.W_y = nn.Sequential(
            conv1x1(F_2, F_3),
            nn.BatchNorm2d(F_3)
        )
        self.active = nn.Sequential(
            conv1x1(F_3, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = self.W_x(x)
        y = self.W_y(y)
        attmap = self.relu(x + y)
        attmap = self.active(attmap)

        return attmap


class CAmodule(nn.Module):
    # Channel-wise attention module
    def __init__(self, in_channels, middle_channels, out_channels):
        super(CAmodule, self).__init__()

        self.ca = CALayer(in_channels)
        self.conv1 = conv3x3(in_channels, middle_channels)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(middle_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        x = self.ca(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SCOATNet(nn.Module):
    def __init__(self, n_channel, n_classes=3, deep_supervision=False,
                 blockname='Res', **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        switch = {
            'VGG': VGGBlock, 'Res': ResBlock
        }
        BasicBlock = switch.get(blockname, VGGBlock)
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = BasicBlock(n_channel, nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = BasicBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = BasicBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = CAmodule(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = CAmodule(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = CAmodule(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = BasicBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = CAmodule(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = CAmodule(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = BasicBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = CAmodule(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = BasicBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.att1_1 = SAmodule(nb_filter[1], nb_filter[2], nb_filter[1])
        self.att2_1 = SAmodule(nb_filter[2], nb_filter[3], nb_filter[2])
        self.att3_1 = SAmodule(nb_filter[3], nb_filter[4], nb_filter[3])
        self.att1_2 = SAmodule(nb_filter[1], nb_filter[2], nb_filter[1])
        self.att2_2 = SAmodule(nb_filter[2], nb_filter[3], nb_filter[2])
        self.att1_3 = SAmodule(nb_filter[1], nb_filter[2], nb_filter[1])

        if self.deep_supervision:
            self.final1 = conv3x3(nb_filter[0], n_classes)
            self.final2 = conv3x3(nb_filter[0], n_classes)
            self.final3 = conv3x3(nb_filter[0], n_classes)
            self.final4 = conv3x3(nb_filter[0], n_classes)
        else:
            self.final = conv3x3(nb_filter[0], n_classes)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        attmap = self.att1_1(x1_0, self.up(x2_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0) + attmap * self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        attmap = self.att2_1(x2_0, self.up(x3_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0) + attmap * self.up(x3_0)], 1))
        attmap = self.att1_2(x1_1, self.up(x2_1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1) + attmap * self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        attmap = self.att3_1(x3_0, self.up(x4_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0) + attmap * self.up(x4_0)], 1))
        attmap = self.att2_2(x2_1, self.up(x3_1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1) + attmap * self.up(x3_1)], 1))
        attmap = self.att1_3(x1_2, self.up(x2_2))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2) + attmap * self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


if __name__ == "__main__":
    model = SCOATNet(1, 1)
    data = torch.ones([2, 1, 224, 224])
    out = model(data)
    print(out.shape)
