"""
Custom Norm operators to enable sync BN, regular BN and for weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F




def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = torch.nn.BatchNorm2d
    normalization_layer = layer(in_channels)
    return normalization_layer


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


class AlignedModule(nn.Module):

    def __init__(self, inplanel,inplaneh, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplaneh, outplane, 1, bias=False)
        
        self.down_l = nn.Conv2d(inplanel, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x1,x2):
        low_feature=x1
        h_feature = x2
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
       
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModulev2(nn.Module):

    def __init__(self, inplanel,inplaneh, outplane, kernel_size=3):
        super(AlignedModulev2, self).__init__()
        self.down_h = nn.Conv2d(inplaneh, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplanel, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(outplane * 2, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        low_feature=x1
        h_feature = x2
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :] , flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        flow_gates = self.flow_gate(torch.cat([h_feature, l_feature], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModulev2PoolingAtten(nn.Module):

    def __init__(self, inplanel,inplaneh, outplane, kernel_size=3):
        super(AlignedModulev2PoolingAtten, self).__init__()
        self.down_h = nn.Conv2d(inplaneh, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplanel, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1,x2):
        low_feature=x1
        h_feature = x2
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output
