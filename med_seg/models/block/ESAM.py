import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from pytorch_wavelets import DWTForward

"""
ESAM被用作一个独立的网络模块，其主要功能是通过卷积层和梯度增强来处理输入特征图，目的是增强图像的边缘信息，从而更好地捕捉边缘细节。
通过这种方式，ESAM旨在将边缘信息集成到特征图中，帮助提高分割的精确度。
"""

    
class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, ksize, padding=1, stride=1, act="silu"):
        super().__init__()
        self.dconv = nn.Conv2d(
            in_channels, in_channels, ksize,
            stride=stride,padding=padding, groups=in_channels
        )
 
    def forward(self, x):
    
        x = self.dconv(x)
        return x
    
class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i]//2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)
            
    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


# class ESAM(nn.Module):
#     def __init__(self, in_channels):
#         super(ESAM, self).__init__()
#         # self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv1 = MultiScaleDWConv(in_channels)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 保持通道数不变
#         self.bn = nn.BatchNorm2d(in_channels)  # 用于conv1和conv2的输出
#         self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, in_channels)  # 注意此处

#     def forward(self, x):
#         y = run_sobel(self.sobel_x1, self.sobel_y1, x)
#         y = F.relu(self.bn(y))
#         y = self.conv1(y)
#         y = x + y
#         y = self.conv2(y)
#         y = F.relu(self.bn(y))  # 使用self.bn而不是self.ban

#         return y
    
class ESAM(nn.Module):
    def __init__(self, in_channels,out_channels,stride,drop):
        super(ESAM, self).__init__()
        self.conv1 = MultiScaleDWConv(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=stride)
        self.bn = nn.BatchNorm2d(in_channels)
        self.ban = nn.BatchNorm2d(out_channels)
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, in_channels)
        self.drop = nn.Dropout(drop)
        # self.up=HWD(in_channels,out_channels)
        self.in_channels=in_channels
        self.out_channels=out_channels
    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, x)
        y = F.gelu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = F.gelu(self.bn(y))
        y = self.drop(y)
        y = self.conv2(y)
        y = F.gelu(self.ban(y))

        # if(self.in_channels!=self.out_channels):
        # y = self.up(y)

        return y
    


class Edgenet(nn.Module):
    def __init__(self):
        super(Edgenet, self).__init__()

        
        self.up1 = ESAM(120,240,2,0.5)
        self.up2 = ESAM(240,240,1,0.4)
        self.up3 = ESAM(480,480,1,0.3)
        self.up4 = ESAM(960,960,1,0.2)
        self.up11 = ESAM(240,480,2,0.45)
        self.up22 = ESAM(480,960,2,0.35)
        self.up33 = ESAM(960,960,1,0.25)

    def forward(self, f1,f2,f3,f4 ):

        out1 = self.up1(f1)
        out2 = self.up2(f2)
        out3 = self.up3(f3)
        out4 = self.up4(f4)
        out = out1 + out2
        out = self.up11(out)
        out = out + out3
        out = self.up22(out)
        out = out + out4
        out = self.up33(out)


        return out

    
class ESAM1(nn.Module):
    def __init__(self, in_channels,stride,drop):
        super(ESAM1, self).__init__()
        self.conv1 = MultiScaleDWConv(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1,stride=stride)
        self.bn = nn.BatchNorm2d(in_channels)
        self.ban = nn.BatchNorm2d(in_channels)
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, in_channels)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, x)
        y = F.gelu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = F.gelu(self.ban(y))
        y = self.drop(y)
        y = self.conv2(y)
        y = F.gelu(self.ban(y))
        y = self.drop(y)

        return y
    


class Edgenet1(nn.Module):
    def __init__(self):
        super(Edgenet1, self).__init__()

        
        self.up1 = ESAM1(120,1,0.2)
        self.up2 = ESAM1(240,1,0.3)
        self.up3 = ESAM1(480,1,0.4)
        self.up4 = ESAM1(960,1,0.5)
        self.up11 = ESAM1(240,1,0.25)
        self.up22 = ESAM1(480,1,0.35)
        self.up33 = ESAM1(960,1,0.45)
        self.conv1 = nn.Conv2d(120, 240, kernel_size=1,stride=2)
        self.conv2 = nn.Conv2d(240, 480, kernel_size=1,stride=2)
        self.conv3 = nn.Conv2d(480, 960, kernel_size=1,stride=2)
        self.ba1 = nn.BatchNorm2d(240)
        self.ba2 = nn.BatchNorm2d(480)
        self.ba3 = nn.BatchNorm2d(960)


    def forward(self, f1,f2,f3,f4 ):

        out1 = self.up1(f1)
        out11 = self.conv1(out1)
        out11 = F.gelu(self.ba1(out11))
        out2 = self.up2(f2)
        out3 = self.up3(f3)
        out4 = self.up4(f4)
        out2 = out11 + out2
        out2 = self.up11(out2)
        out22 = self.conv2(out2)
        out22 = F.gelu(self.ba2(out22))
        out3 = out22 + out3
        out3 = self.up22(out3)
        out33 = self.conv3(out3)
        out33 = F.gelu(self.ba3(out33))
        out4 = out33 + out4
        out4 = self.up33(out4)


        return [out1,out2,out3,out4]
