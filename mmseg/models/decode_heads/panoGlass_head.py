# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torchvision
import torch.nn.functional as F

@HEADS.register_module()
class MyHead(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(MyHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.getr = GETR()
        self.cls = nn.Conv2d(32,3,kernel_size=1)
    def forward(self, inputs):
        output = self.getr(inputs)
        output = self.cls(output)
        return output
    
class GETR(nn.Module):

    def __init__(self):

        super().__init__()
        self.rgb_input_proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.fusion = nn.Conv2d(256, 256, 1)
        self.mask_head = MaskHeadSmallConv(512, [2048, 1024, 512, 256], 256)
        self.hdcn_3 = hdcn(2048, hdcnType=2)
        self.hdcn_2 = hdcn(1024, hdcnType=2)
        self.hdcn_1 = hdcn(512, hdcnType=1)
        self.hdcn_0 = hdcn(256, hdcnType=1)
        self.hdcn_channel_3 = nn.Conv2d(512, 256, kernel_size=1)
        self.hdcn_channel_1 = nn.Conv2d(128, 256, kernel_size=1)
        self.hdcn_channel_0 = nn.Conv2d(64, 256, kernel_size=1)
        self.fusion_module_3  = nn.Sequential(nn.Conv2d(768, 256, 1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.fusion_module_2  = nn.Sequential(nn.Conv2d(768, 256, 1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.fusion_module_1  = nn.Sequential(nn.Conv2d(768, 128, 1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.fusion_module_0  = nn.Sequential(nn.Conv2d(768, 64, 1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        nn.init.kaiming_uniform_(self.hdcn_channel_3.weight, a=1)
        nn.init.kaiming_uniform_(self.hdcn_channel_1.weight, a=1)
        nn.init.kaiming_uniform_(self.hdcn_channel_0.weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_3[0].weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_2[0].weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_0[0].weight, a=1)

    def forward(self, inputs):
        """Forward function."""
        fusion_proj = self.rgb_input_proj(inputs[3])
        fusion_memory = self.fusion(fusion_proj)
        rgb_features_hdcn_3 = self.hdcn_3(inputs[3])
        rgb_features_hdcn_2 = self.hdcn_2(inputs[2])
        rgb_features_hdcn_1 = self.hdcn_1(inputs[1])
        rgb_features_hdcn_0 = self.hdcn_0(inputs[0])
        rgb_features_hdcn_256channel_3 = self.hdcn_channel_3(rgb_features_hdcn_3)
        rgb_features_hdcn_256channel_1 = self.hdcn_channel_1(rgb_features_hdcn_1)
        rgb_features_hdcn_256channel_0 = self.hdcn_channel_0(rgb_features_hdcn_0)
        
        fusion_memory_3 = self.fusion_module_3(torch.cat(
            (F.interpolate(rgb_features_hdcn_2, size=inputs[3].shape[-2:], mode="bilinear"), 
             F.interpolate(rgb_features_hdcn_256channel_1, size=inputs[3].shape[-2:], mode="bilinear"), 
             F.interpolate(rgb_features_hdcn_256channel_0, size=inputs[3].shape[-2:], mode="bilinear")),1))
        fusion_memory_2 = self.fusion_module_2(torch.cat(
            (F.interpolate(rgb_features_hdcn_256channel_3, size=inputs[2].shape[-2:], mode="bilinear"), 
             F.interpolate(rgb_features_hdcn_256channel_1, size=inputs[2].shape[-2:], mode="bilinear"), 
             F.interpolate(rgb_features_hdcn_256channel_0, size=inputs[2].shape[-2:], mode="bilinear")),1))
        fusion_memory_1 = self.fusion_module_1(torch.cat(
            (F.interpolate(rgb_features_hdcn_256channel_3, size=inputs[1].shape[-2:], mode="bilinear"), 
             F.interpolate(rgb_features_hdcn_2, size=inputs[1].shape[-2:], mode="bilinear"), 
             F.interpolate(rgb_features_hdcn_256channel_0, size=inputs[1].shape[-2:], mode="bilinear")),1))
        fusion_memory_0 = self.fusion_module_0(torch.cat(
            (F.interpolate(rgb_features_hdcn_256channel_3, size=inputs[0].shape[-2:], mode="bilinear"), 
             F.interpolate(rgb_features_hdcn_2, size=inputs[0].shape[-2:], mode="bilinear"), 
             F.interpolate(rgb_features_hdcn_256channel_1, size=inputs[0].shape[-2:], mode="bilinear")),1))
        
        output = self.mask_head(fusion_memory,
                                [inputs[3], inputs[2],
                                 inputs[1], inputs[0]],
                                [fusion_memory_3, fusion_memory_2,
                                fusion_memory_1, fusion_memory_0])
        return output

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        # original
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # max
        # torch.max(x, 1)[0].unsqueeze(1)
        # avg
        return torch.mean(x, 1).unsqueeze(1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

    
class hdcn(nn.Module):
    def __init__(self, input_channels, dr1=1, dr2=2, dr3=3, dr4=4, hdcnType = 1):
        super(hdcn, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)
        self.channels_double = int(input_channels / 2)
        self.dr1 = dr1
        self.dr2 = dr2
        self.dr3 = dr3
        self.dr4 = dr4
        self.padding1 = 1 * dr1
        self.padding2 = 2 * dr2
        self.padding3 = 3 * dr3
        self.padding4 = 4 * dr4
        self.type = hdcnType
        if self.type == 1: 
            self.p1_2_channel_reduction = nn.Sequential(
                nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
                nn.BatchNorm2d(self.channels_single), nn.ReLU()
            )

            self.p1_d1_offset1 = nn.Conv2d(self.channels_single, 2*3*1,(3,1),1,padding=(self.padding1,0),dilation=(self.dr1,1))
            self.p1_d1_mask1 = nn.Conv2d(self.channels_single, 3*1,(3,1),1,padding=(self.padding1,0),dilation=(self.dr1,1))
            #self.p1_d1_weight1 = nn.Parameter(torch.Tensor(np.zeros([self.channels_single, self.channels_single, 3, 1])))
            self.p1_d1_weight1 = nn.Parameter(torch.empty(self.channels_single,self.channels_single,3,1))
            nn.init.kaiming_uniform_(self.p1_d1_weight1,a=1)
            self.p1_d1_conv2 = nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, self.padding1),dilation=(1, self.dr1))

            self.p1_d1_bachNorm = nn.BatchNorm2d(self.channels_single)
            self.p1_d1_ReLU = nn.ReLU()

            self.p1_d2_conv1 = nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, self.padding1),dilation=(1, self.dr1))

            
            self.p1_d2_offset2 = nn.Conv2d(self.channels_single, 2*3*1,(3,1),1,padding=(self.padding1,0),dilation=(self.dr1,1))
            self.p1_d2_mask2 = nn.Conv2d(self.channels_single, 3*1,(3,1),1,padding=(self.padding1,0),dilation=(self.dr1,1))
            #self.p1_d2_weight2 = nn.Parameter(torch.Tensor(np.zeros([self.channels_single, self.channels_single, 3, 1])))
            self.p1_d2_weight2 = nn.Parameter(torch.empty(self.channels_single, self.channels_single, 3, 1))
            nn.init.kaiming_uniform_(self.p1_d2_weight2,a=1)
            self.p1_d2_bachNorm = nn.BatchNorm2d(self.channels_single)
            self.p1_d2_ReLU = nn.ReLU()
            
            self.p1_fusion = nn.Sequential(
                nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                nn.BatchNorm2d(self.channels_single), nn.ReLU()
            )

            self.p2_d1_offset1 = nn.Conv2d(self.channels_double, 2*5*1,(5,1),1,padding=(self.padding2,0),dilation=(self.dr2,1))
            self.p2_d1_mask1 = nn.Conv2d(self.channels_double, 5*1,(5,1),1,padding=(self.padding2,0),dilation=(self.dr2,1))
            #self.p2_d1_weight1 = nn.Parameter(torch.Tensor(np.zeros([self.channels_single, self.channels_double, 5, 1])))
            self.p2_d1_weight1 = nn.Parameter(torch.empty(self.channels_single, self.channels_double, 5, 1))
            nn.init.kaiming_uniform_(self.p2_d1_weight1,a=1)
            self.p2_d1_conv2 = nn.Conv2d(self.channels_single, self.channels_single, (1, 5), 1, padding=(0, self.padding2),dilation=(1, self.dr2))

            self.p2_d1_bachNorm = nn.BatchNorm2d(self.channels_single)
            self.p2_d1_ReLU = nn.ReLU()
            self.p2_d2_conv1 = nn.Conv2d(self.channels_double, self.channels_single, (1, 5), 1, padding=(0, self.padding2),dilation=(1, self.dr2))

            
            self.p2_d2_offset2 = nn.Conv2d(self.channels_single, 2*5*1,(5,1),1,padding=(self.padding2,0),dilation=(self.dr2,1))
            self.p2_d2_mask2 = nn.Conv2d(self.channels_single, 5*1,(5,1),1,padding=(self.padding2,0),dilation=(self.dr2,1))
            #self.p2_d2_weight2 = nn.Parameter(torch.Tensor(np.zeros([self.channels_single, self.channels_single, 5, 1])))
            self.p2_d2_weight2 = nn.Parameter(torch.empty(self.channels_single, self.channels_single, 5, 1))
            nn.init.kaiming_uniform_(self.p2_d2_weight2,a=1)
            self.p2_d2_bachNorm = nn.BatchNorm2d(self.channels_single)
            self.p2_d2_ReLU = nn.ReLU()
            
            self.p2_fusion = nn.Sequential(
                nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                nn.BatchNorm2d(self.channels_single), nn.ReLU()
            )
        elif self.type == 2:
            self.p3_4_channel_reduction = nn.Sequential(
                nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
                nn.BatchNorm2d(self.channels_single), nn.ReLU()
            )
            self.p3_d1_offset1 = nn.Conv2d(self.channels_single, 2*7*1,(7,1),1,padding=(self.padding3,0),dilation=(self.dr3,1))
            self.p3_d1_mask1 = nn.Conv2d(self.channels_single, 7*1,(7,1),1,padding=(self.padding3,0),dilation=(self.dr3,1))
            #self.p3_d1_weight1 = nn.Parameter(torch.Tensor(np.zeros([self.channels_single, self.channels_single, 7, 1])))
            self.p3_d1_weight1 = nn.Parameter(torch.empty(self.channels_single, self.channels_single, 7, 1))
            nn.init.kaiming_uniform_(self.p3_d1_weight1,a=1)
            self.p3_d1_conv2 = nn.Conv2d(self.channels_single, self.channels_single, (1, 7), 1, padding=(0, self.padding3),dilation=(1, self.dr3))

            self.p3_d1_bachNorm = nn.BatchNorm2d(self.channels_single)
            self.p3_d1_ReLU = nn.ReLU()

            self.p3_d2_conv1 = nn.Conv2d(self.channels_single, self.channels_single, (1, 7), 1, padding=(0, self.padding3),dilation=(1, self.dr3))
            self.p3_d2_offset2 = nn.Conv2d(self.channels_single, 2*7*1,(7,1),1,padding=(self.padding3,0),dilation=(self.dr3,1))
            self.p3_d2_mask2 = nn.Conv2d(self.channels_single, 7*1,(7,1),1,padding=(self.padding3,0),dilation=(self.dr3,1))
            #self.p3_d2_weight2 = nn.Parameter(torch.Tensor(np.zeros([self.channels_single, self.channels_single, 7, 1])))
            self.p3_d2_weight2 = nn.Parameter(torch.empty(self.channels_single, self.channels_single, 7, 1))
            nn.init.kaiming_uniform_(self.p3_d2_weight2,a=1)
            self.p3_d2_bachNorm = nn.BatchNorm2d(self.channels_single)
            self.p3_d2_ReLU = nn.ReLU()
            
            self.p3_fusion = nn.Sequential(
                nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                nn.BatchNorm2d(self.channels_single), nn.ReLU()
            )

            self.p4_d1_offset1 = nn.Conv2d(self.channels_double, 2*9*1,(9,1),1,padding=(self.padding4,0),dilation=(self.dr4,1))
            self.p4_d1_mask1 = nn.Conv2d(self.channels_double, 9*1,(9,1),1,padding=(self.padding4,0),dilation=(self.dr4,1))
            #self.p4_d1_weight1 = nn.Parameter(torch.Tensor(np.zeros([self.channels_single, self.channels_double, 9, 1])))
            self.p4_d1_weight1 = nn.Parameter(torch.empty(self.channels_single, self.channels_double, 9, 1))
            nn.init.kaiming_uniform_(self.p4_d1_weight1,a=1)
            self.p4_d1_conv2 = nn.Conv2d(self.channels_single, self.channels_single, (1, 9), 1, padding=(0, self.padding4),dilation=(1, self.dr4))
            self.p4_d1_bachNorm = nn.BatchNorm2d(self.channels_single)
            self.p4_d1_ReLU = nn.ReLU()

            self.p4_d2_conv1 = nn.Conv2d(self.channels_double, self.channels_single, (1, 9), 1, padding=(0, self.padding4),dilation=(1, self.dr4))
            self.p4_d2_offset2 = nn.Conv2d(self.channels_single, 2*9*1,(9,1),1,padding=(self.padding4, 0),dilation=(self.dr4,1))
            self.p4_d2_mask2 = nn.Conv2d(self.channels_single, 9*1,(9,1),1,padding=(self.padding4, 0),dilation=(self.dr4,1))
            #self.p4_d2_weight2 = nn.Parameter(torch.Tensor(np.zeros([self.channels_single, self.channels_single, 9, 1])))
            self.p4_d2_weight2 = nn.Parameter(torch.empty(self.channels_single,self.channels_single,9,1))
            nn.init.kaiming_uniform_(self.p4_d2_weight2,a=1)
            self.p4_d2_bachNorm = nn.BatchNorm2d(self.channels_single)
            self.p4_d2_ReLU = nn.ReLU()
            
            self.p4_fusion = nn.Sequential(
                nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                nn.BatchNorm2d(self.channels_single), nn.ReLU()
            )
        self.cbam = CBAM(self.channels_double)

        self.channel_reduction = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                #nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        if self.type == 1:
            p1_2_input = self.p1_2_channel_reduction(x)
            p1_d1_1 = torchvision.ops.deform_conv2d(
                input=p1_2_input, 
                offset=self.p1_d1_offset1(p1_2_input),
                weight = self.p1_d1_weight1, 
                mask=self.p1_d1_mask1(p1_2_input), 
                padding=(self.padding1,0), 
                dilation=(self.dr1,1)
            )

            p1_d1_2 = self.p1_d1_conv2(p1_d1_1)
            p1_d2_1 = self.p1_d2_conv1(p1_2_input)
            p1_d2_2 = torchvision.ops.deform_conv2d(
                input=p1_d2_1, 
                offset=self.p1_d2_offset2(p1_d2_1), 
                weight = self.p1_d2_weight2,
                mask=self.p1_d2_mask2(p1_d2_1),
                padding=(self.padding1,0), 
                dilation=(self.dr1,1)
            )
            p1 = self.p1_fusion(torch.cat((self.p1_d1_ReLU(self.p1_d1_bachNorm(p1_d1_2)), self.p1_d2_ReLU(self.p1_d2_bachNorm(p1_d2_2))), 1))            
            p2_d1_1 = torchvision.ops.deform_conv2d(
                input=torch.cat((p1_2_input, p1), 1), 
                offset=self.p2_d1_offset1(torch.cat((p1_2_input, p1), 1)),
                weight = self.p2_d1_weight1, 
                mask=self.p2_d1_mask1(torch.cat((p1_2_input, p1), 1)), 
                padding=(self.padding2,0), 
                dilation=(self.dr2,1)
            )

            p2_d1_2 = self.p2_d1_conv2(p2_d1_1)
            p2_d2_1 = self.p2_d2_conv1(torch.cat((p1_2_input, p1), 1))
            p2_d2_2 = torchvision.ops.deform_conv2d(
                input=p2_d2_1, 
                offset=self.p2_d2_offset2(p2_d2_1), 
                weight = self.p2_d2_weight2,
                mask=self.p2_d2_mask2(p2_d2_1),
                padding=(self.padding2,0), 
                dilation=(self.dr2,1)
            )
            p2 = self.p2_fusion(torch.cat((self.p2_d1_ReLU(self.p2_d1_bachNorm(p2_d1_2)), self.p2_d2_ReLU(self.p2_d2_bachNorm(p2_d2_2))), 1))
            channel_reduction = self.channel_reduction(self.cbam(torch.cat((p1, p2), 1)))
        elif self.type == 2:
            p3_4_input = self.p3_4_channel_reduction(x)
            p3_d1_1 = torchvision.ops.deform_conv2d(
                input=p3_4_input, 
                offset=self.p3_d1_offset1(p3_4_input),
                weight = self.p3_d1_weight1, 
                mask=self.p3_d1_mask1(p3_4_input), 
                padding=(self.padding3,0), 
                dilation=(self.dr3,1)
            )

            p3_d1_2 = self.p3_d1_conv2(p3_d1_1)
            p3_d2_1 = self.p3_d2_conv1(p3_4_input)
            p3_d2_2 = torchvision.ops.deform_conv2d(
                input=p3_d2_1, 
                offset=self.p3_d2_offset2(p3_d2_1), 
                weight = self.p3_d2_weight2,
                mask=self.p3_d2_mask2(p3_d2_1),
                padding=(self.padding3,0), 
                dilation=(self.dr3,1)
            )
            p3_d1_2_bach = self.p3_d1_ReLU(self.p3_d1_bachNorm(p3_d1_2))
            p3_d2_2_bach = self.p3_d2_ReLU(self.p3_d2_bachNorm(p3_d2_2))
            p3 = self.p3_fusion(torch.cat((p3_d1_2_bach, p3_d2_2_bach),1))           
            p4_d1_1 = torchvision.ops.deform_conv2d(
                input=torch.cat((p3_4_input, p3),1), 
                offset=self.p4_d1_offset1(torch.cat((p3_4_input, p3),1)),
                weight = self.p4_d1_weight1, 
                mask=self.p4_d1_mask1(torch.cat((p3_4_input, p3),1)), 
                padding=(self.padding4,0), 
                dilation=(self.dr4,1)
            )

            p4_d1_2 = self.p4_d1_conv2(p4_d1_1)
            p4_d2_1 = self.p4_d2_conv1(torch.cat((p3_4_input, p3),1))
            p4_d2_2 = torchvision.ops.deform_conv2d(
                input=p4_d2_1, 
                offset=self.p4_d2_offset2(p4_d2_1), 
                weight = self.p4_d2_weight2,
                mask=self.p4_d2_mask2(p4_d2_1),
                padding=(self.padding4,0), 
                dilation=(self.dr4,1)
            )
            p4 = self.p4_fusion(torch.cat((self.p4_d1_ReLU(self.p4_d1_bachNorm(p4_d1_2)), self.p4_d2_ReLU(self.p4_d2_bachNorm(p4_d2_2))), 1))
            channel_reduction = self.channel_reduction(self.cbam(torch.cat((p3, p4), 1)))
        return channel_reduction
    
class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        # [512, 256, 128, 64, 32, 16]
        inter_dims = [dim, context_dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16]
        self.inference_module1 = InferenceModule(inter_dims[1], inter_dims[1])
        self.inference_module2 = InferenceModule(inter_dims[1], inter_dims[2])
        self.inference_module3 = InferenceModule(inter_dims[2], inter_dims[3])
        self.inference_module4 = InferenceModule(inter_dims[3], inter_dims[4])
        self.inference_module5 = InferenceModule(inter_dims[4], inter_dims[5])

        self.rgb_adapter1 = nn.Sequential(nn.Conv2d(2048, inter_dims[1], 1),
                                          nn.BatchNorm2d(inter_dims[1]),
                                          nn.ReLU(inplace=True))
        self.rgb_adapter2 = nn.Sequential(nn.Conv2d(1024, inter_dims[1], 1),
                                          nn.BatchNorm2d(inter_dims[1]),
                                          nn.ReLU(inplace=True))
        self.rgb_adapter3 = nn.Sequential(nn.Conv2d(512, inter_dims[2], 1),
                                          nn.BatchNorm2d(inter_dims[2]),
                                          nn.ReLU(inplace=True))
        self.rgb_adapter4 = nn.Sequential(nn.Conv2d(256, inter_dims[3], 1),
                                          nn.BatchNorm2d(inter_dims[3]),
                                          nn.ReLU(inplace=True))
        
        self.pa_module1 = PixelAttention(inter_dims[1], 3)
        self.pa_module2 = PixelAttention(inter_dims[1], 3)
        self.pa_module3 = PixelAttention(inter_dims[2], 3)
        self.pa_module4 = PixelAttention(inter_dims[3], 3)

        self.mask_out_conv = nn.Conv2d(5, 1, 1)
        self.edge_out_conv = nn.Conv2d(5, 1, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fusion_memory, rgb, rgb_hdcn):

        x = fusion_memory

        rgb_for_infer = self.rgb_adapter1(rgb[0])

        x = self.inference_module1(self.pa_module1(x, rgb_for_infer, rgb_hdcn[0]))  # 1/32

        rgb_for_infer = self.rgb_adapter2(rgb[1])

        x = F.interpolate(x, size=rgb_for_infer.shape[-2:], mode="bilinear")
        x = self.inference_module2(self.pa_module2(x, rgb_for_infer, rgb_hdcn[1]))  # 1/16

        rgb_for_infer = self.rgb_adapter3(rgb[2])

        x = F.interpolate(x, size=rgb_for_infer.shape[-2:], mode="bilinear")
        x = self.inference_module3(self.pa_module3(x, rgb_for_infer, rgb_hdcn[2]))  # 1/8

        rgb_for_infer = self.rgb_adapter4(rgb[3])

        x = F.interpolate(x, size=rgb_for_infer.shape[-2:], mode="bilinear")
        x = self.inference_module4(self.pa_module4(x, rgb_for_infer, rgb_hdcn[3]))  # 1/4

        return x


class PixelAttention(nn.Module):
    def __init__(self, inchannels, times):
        super(PixelAttention, self).__init__()

        self.mask_conv1 = nn.Sequential(nn.Conv2d(inchannels * 2, inchannels, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm2d(inchannels),
                                       nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm2d(inchannels),
                                       nn.Conv2d(inchannels, 1, 1))
        self.mask_conv2 = nn.Sequential(nn.Conv2d(inchannels * 2, inchannels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(inchannels),
                                        nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(inchannels),
                                        nn.Conv2d(inchannels, 1, 1))

    def forward(self, x, rgb, hdcn):

        mask1 = self.mask_conv1(torch.cat([x, rgb], 1))
        mask1 = torch.sigmoid(mask1)
        rx = x + rgb * mask1
        mask2 = self.mask_conv2(torch.cat([x, hdcn], 1))
        mask2 = torch.sigmoid(mask2)
        
        x = rx + hdcn * mask2

        return x


class InferenceModule(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, indim, outdim):
        super().__init__()

        self.conv_block = nn.Sequential(nn.Conv2d(indim, outdim, 3, padding=1),
                                        nn.BatchNorm2d(outdim),
                                        nn.ReLU(inplace=True))

        self.edge_conv = nn.Sequential(nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True))
        self.mask_conv = nn.Sequential(nn.Conv2d(outdim * 2, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv_block(x)
        edge_feature = self.edge_conv(x)
        x = self.mask_conv(torch.cat([edge_feature, x], 1))
        return x