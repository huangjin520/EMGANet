import torch.nn as nn

from models.block.Base import Conv3Relu


# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         inter_channels = in_channels // 4
        
#         self.head = nn.Sequential(Conv3Relu(in_channels, inter_channels),
#                                   nn.Dropout(0.5),  # 使用0.1的dropout
#                                   nn.Conv2d(inter_channels, out_channels, (1, 1)))

#     def forward(self, x):
#         # print(x.size())

#         return self.head(x)
    
class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, bn_momentum=0.0003):
        super().__init__()
        inter_channels = in_channels // 4


        self.last_conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(inter_channels, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(inter_channels, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )


        self.classify = nn.Conv2d(in_channels=inter_channels, out_channels= out_channels, kernel_size=1,
                                        stride=1, padding=0, dilation=1, bias=True)

    def forward(self, x):
       
        x = self.last_conv(x)
        pred = self.classify(x)
        return pred