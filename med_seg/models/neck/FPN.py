import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.Base import Conv3Relu
from models.block.Drop import DropBlock
from models.block.Field import PPM, ASPP, SPP
from models.neck.SCN import AlignedModule,AlignedModulev2PoolingAtten

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            normal_layer(out_planes),
            nn.ReLU(inplace=True),
    )

class FPNNeck(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 1, inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 2, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 4, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 8, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)
        
        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)
        
        self.scn41= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn31= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn21= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.final_Conv5 = Conv3Relu(inplanes , inplanes)
        

        # PPM/ASPP比SPP好
        if "+ppm+" in neck_name:
            self.expand_field = PPM(inplanes * 8)
        elif "+aspp+" in neck_name:
            self.expand_field = ASPP(inplanes * 8)
        elif "+spp+" in neck_name:
            self.expand_field = SPP(inplanes * 8)
        else:
            self.expand_field = None

        if "fuse" in neck_name:
            self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   # 降维
            self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
            self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)

            self.final_Conv = Conv3Relu(inplanes * 4, inplanes)
            

            self.fuse = True
        else:
            self.fuse = False

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4 = ms_feats
        feature1_h, feature1_w = fa1.size(2), fa1.size(3)

        [fa1, fa2, fa3, fa4] = self.drop([fa1, fa2, fa3, fa4])  # dropblock

        feature1 = self.stage1_Conv1(torch.cat([fa1], 1))  # inplanes
        feature2 = self.stage2_Conv1(torch.cat([fa2], 1))  # inplanes * 2
        feature3 = self.stage3_Conv1(torch.cat([fa3], 1))  # inplanes * 4
        feature4 = self.stage4_Conv1(torch.cat([fa4], 1))  # inplanes * 8
        # print(feature1.size())
        if self.expand_field is not None:
            feature4 = self.expand_field(feature4)
       
        feature3_2 = self.stage4_Conv_after_up(self.up(feature4))
        feature3 = self.stage3_Conv2(torch.cat([feature3, feature3_2], 1))

        feature2_2 = self.stage3_Conv_after_up(self.up(feature3))
        feature2 = self.stage2_Conv2(torch.cat([feature2, feature2_2], 1))

        feature1_2 = self.stage2_Conv_after_up(self.up(feature2))
        feature1 = self.stage1_Conv2(torch.cat([feature1, feature1_2], 1))
        

        if self.fuse:
           
            feature4=self.scn41(feature1, self.stage4_Conv3(feature4))
            feature3=self.scn31(feature1, self.stage3_Conv3(feature3))
            feature2=self.scn21(feature1, self.stage2_Conv3(feature2))

            [feature1, feature2, feature3, feature4] = self.drop([feature1, feature2, feature3, feature4])  # dropblock

            feature = self.final_Conv(torch.cat([feature1, feature2, feature3, feature4], 1))
           
        else:
            feature = feature1
            feature=self.final_Conv5(feature1)

        return feature

