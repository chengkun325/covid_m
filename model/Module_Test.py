import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    conv_bn_relu = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
    return conv_bn_relu

class MultiScaleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MultiScaleConv, self).__init__()
        assert in_channel % 4 == 0, "The num of in_channel can not be divided by 4."
        
        self.single_ch = in_channel // 4
        # self.expansion = out_channel // in_channel
        
        self.conv3_3 = nn.Conv2d(self.single_ch, 2 * self.single_ch, kernel_size=3, stride=1, padding=1)
        
        self.atrous_conv3 = nn.Conv2d(self.single_ch, 2 * self.single_ch, 3, 1, padding=3, dilation=3)
        self.atrous_conv5 = nn.Conv2d(2 * self.single_ch, 2 * self.single_ch, 3, 1, padding=5, dilation=5)
        self.atrous_conv7 = nn.Conv2d(2 * self.single_ch, self.single_ch, 3, 1, padding=7, dilation=7)
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
        x0, x1, x2, x3 = torch.split(x, self.single_ch, dim=1)
        
        x0 = self.conv3_3(x0)
        x1 = self.atrous_conv3(x1)
        x1, x2_tmp = torch.split(x1, self.single_ch, dim=1)
        x2 = self.atrous_conv5(torch.cat([x2, x2_tmp], dim=1))
        x2, x3_tmp = torch.split(x2, self.single_ch, dim=1)
        x3 = self.atrous_conv7(torch.cat([x3, x3_tmp], dim=1))
        
        x = torch.cat([x0, x1, x2, x3], dim=1)
        
        return x
        

if __name__ == "__main__":
    # if 1 != 3:
    # assert 1 == 3, "fjdkls"
    m = MultiScaleConv(32, 32).cuda()
    img = torch.randn(8, 32, 16, 16).cuda()
    y = m(img)
    print(y.size())
    
# end main