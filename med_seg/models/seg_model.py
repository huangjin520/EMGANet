import os
import re
from copy import deepcopy

from thop import profile
from thop import clever_format
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.swin_transformer_v2 import SwinTransformerV2, swin_transformer_v2_l,swin_transformer_v2_b
from models.block.Base import ChannelChecker
from models.head.FCN import FCNHead
from models.neck.FPN import FPNNeck
from collections import OrderedDict
from util.common import ScaleInOutput
from models.backbone.groupmixformer import GroupMixFormer
from models.block.ESAM import Edgenet
from models.block.SKFusion import SKFusion
from models.neck.EMCAD import EMCAD

SwinTransformerV2=swin_transformer_v2_b(in_channels=3,
                 input_resolution= [256, 256],
                 window_size= 16,
                 patch_size= 4,
                 ff_feature_ratio=4)


class Seg_Detection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1]))  # backbone的名称中必须在"_"之后加上它的通道数
        self.inplanes=120
        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)
        self._create_heads(opt.head)

        self.GMA=GroupMixFormer(embedding_dims=[120,240,480,960],
                                serial_depths=[8, 8, 12, 8], mlp_ratios=[2, 2, 4, 4],
                                drop_path_rate=0.5)
        self.Edgenet=Edgenet()
        self.decoder = EMCAD(channels=[960,480,240,120], kernel_sizes=[1,3,5], 
                             expansion_factor=2, dw_parallel=True,
                             add=True, lgag_ks=3, activation='relu')
        self.fusion = SKFusion(960)

        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)   # todo:这里预训练初始化和 hrnet主干网络的初始化有冲突，必须要改！


    def forward(self, x):
        _, _, h_input, w_input = x.shape
        
        # encoder
        f1, f2, f3, f4 = self.GMA(x)  # feature_a_1: 输入图像a的最大输出特征图
        
        # edge and fusion
        d4=self.Edgenet(f1,f2,f3,f4)
        f4=self.fusion([d4,f4])
        
        # decoder1
        # ms_feats = f1, f2, f3, f4  # 多尺度特征
        # feature = self.neck(ms_feats)
        
        # decoder2
        dec_outs = self.decoder(f4, [f3, f2, f1])
        feature=dec_outs[3]

        # head
        out = self.head_forward(feature , out_size=(h_input, w_input))

        return out



    def head_forward(self, feature , out_size):


        out = F.interpolate(self.head(feature ), size=out_size, mode='bilinear', align_corners=True)


        return out


    def _init_weight(self, pretrain=''):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 只要是卷积都操作，都对weight和bias进行kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  # bn层都权重初始化为1， bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                                if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))


    def _create_backbone(self, backbone):

        if 'swinv2' in backbone:
            self.backbone = SwinTransformerV2

        else:
            raise Exception('Not Implemented yet: {}'.format(backbone))

    def _create_neck(self, neck):
        if 'fpn' in neck:
            self.neck = FPNNeck(self.inplanes, neck)

    def _select_head(self, head):
        if head == 'fcn':
            return FCNHead(self.inplanes, 2)


    def _create_heads(self, head):
        self.head = self._select_head(head)


       

# if __name__=='__main__':
    
#     img=torch.randn(2,3,256,256)
    
#     import argparse
#     parser = argparse.ArgumentParser('Seg Detection train')
#     parser.add_argument("--backbone", type=str, default="cswin_s_64")
#     parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
#     parser.add_argument("--head", type=str, default="fcn")
#     parser.add_argument("--loss", type=str, default="bce+dice")
#     parser.add_argument("--pretrain", type=str,
#                         default=" ")  # 预训练权重路径
#     parser.add_argument("--input-size", type=int, default=448)
    
#     opt = parser.parse_args()
#     model=Seg_Detection(opt)
#     pred=model(img)
#     print(pred.size())
   
