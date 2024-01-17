from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from hr_layers import *
from layers import upsample


class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        
        # decoder
        self.convs = nn.ModuleDict()
        
        # adaptive block
        if self.num_ch_dec[0] < 16:
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
        
        # adaptive block
            self.convs["72"] = Attention_Module(2 * self.num_ch_dec[4],  2 * self.num_ch_dec[4]  , self.num_ch_dec[4])
            self.convs["36"] = Attention_Module(self.num_ch_dec[4], 3 * self.num_ch_dec[3], self.num_ch_dec[3])
            self.convs["18"] = Attention_Module(self.num_ch_dec[3], self.num_ch_dec[2] * 3 + 64 , self.num_ch_dec[2])
            self.convs["9"] = Attention_Module(self.num_ch_dec[2], 64, self.num_ch_dec[1])
        else: 
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0] + 3,self.num_ch_dec[0])
            self.convs["72"] = Attention_Module(self.num_ch_enc[4] + 3, self.num_ch_enc[3] * 2, 256)
            self.convs["36"] = Attention_Module(256 + 3, self.num_ch_enc[2] * 3, 128)
            self.convs["18"] = Attention_Module(128 + 3, self.num_ch_enc[1] * 3 + 64 , 64)
            self.convs["9"] = Attention_Module(64 + 3, 64, 32)
        for i in range(5):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.convs["consist1"] = Consist_another(self.num_ch_dec[3] + self.num_ch_dec[4] + 3, self.num_ch_dec[3])
        self.convs["consist2"] = Consist_another(self.num_ch_dec[2] + self.num_ch_dec[3] + 3, self.num_ch_dec[2])
        self.convs["consist3"] = Consist_another(self.num_ch_dec[1] + self.num_ch_dec[2] + 3, self.num_ch_dec[1])
        self.convs["consist4"] = Consist_another(self.num_ch_dec[0] + self.num_ch_dec[1] + 3, self.num_ch_dec[0])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.High_Frequency_Upsample = High_Frequency_Upsample(upscale_factor=2)
        self.ASPP = ASPP_another(self.num_ch_enc[4])
        self.Conv1x1 = Conv1x1(self.num_ch_enc[4], self.num_ch_enc[4])

    def forward(self, imgs, input_features):

        rgb_down2 = F.interpolate(imgs, scale_factor = 0.5, mode='bilinear', align_corners=False)
        rgb_down4 = F.interpolate(rgb_down2, scale_factor = 0.5, mode='bilinear', align_corners=False)
        rgb_down8 = F.interpolate(rgb_down4, scale_factor = 0.5, mode='bilinear', align_corners=False)
        rgb_down16 = F.interpolate(rgb_down8, scale_factor = 0.5, mode='bilinear', align_corners=False)
        rgb_down32 = F.interpolate(rgb_down16, scale_factor = 0.5, mode='bilinear', align_corners=False)
        rgb_up16 = F.interpolate(rgb_down32, rgb_down16.shape[2:], mode='bilinear', align_corners=False)
        rgb_up8 = F.interpolate(rgb_down16, rgb_down8.shape[2:], mode='bilinear', align_corners=False)
        rgb_up4 = F.interpolate(rgb_down8, rgb_down4.shape[2:], mode='bilinear', align_corners=False)
        rgb_up2 = F.interpolate(rgb_down4, rgb_down2.shape[2:], mode='bilinear', align_corners=False)
        rgb_up = F.interpolate(rgb_down2, imgs.shape[2:], mode='bilinear', align_corners=False)
        lap1 = imgs - rgb_up
        lap2 = rgb_down2 - rgb_up2
        lap3 = rgb_down4 - rgb_up4
        lap4 = rgb_down8 - rgb_up8
        lap5 = rgb_down16 - rgb_up16

        b, _, _, _ = imgs.shape
        img = lap1.max(dim=1,keepdim=True)
        img = img[0]
        mean = torch.Tensor([torch.mean(torch.abs(i)) for i in img])
        img = torch.cat([torch.abs(img[i]) >= mean[i] for i in range(b)], dim=0).unsqueeze(dim=1)
  
        img2_2 = F.interpolate(
                    lap2, [192, 640], mode="bilinear", align_corners=False)
        img2 = img2_2.max(dim=1,keepdim=True)
        img2 = img2[0]
        mean = torch.Tensor([torch.mean(torch.abs(i)) for i in img2])
        img2 = torch.cat([torch.abs(img2[i]) >= mean[i] for i in range(b)], dim=0).unsqueeze(dim=1)



        img3_2 = F.interpolate(
                    lap3, [192, 640], mode="bilinear", align_corners=False)
        img3 = img3_2.max(dim=1,keepdim=True)
        img3 = img3[0]
        mean = torch.Tensor([torch.mean(torch.abs(i)) for i in img3])
        img3 = torch.cat([torch.abs(img3[i]) >= mean[i] for i in range(b)], dim=0).unsqueeze(dim=1)


        img4_2 = F.interpolate(
                    lap4, [192, 640], mode="bilinear", align_corners=False)
        img4 = img4_2.max(dim=1,keepdim=True)
        img4 = img4[0]
        mean = torch.Tensor([torch.mean(torch.abs(i)) for i in img4])
        img4 = torch.cat([torch.abs(img4[i]) >= mean[i] for i in range(b)], dim=0).unsqueeze(dim=1)




        outputs = {}
        feature144 = input_features[4]
        feature144 = self.Conv1x1(self.ASPP(feature144))
        feature72 = input_features[3]
        feature36 = input_features[2]
        feature18 = input_features[1]
        feature64 = input_features[0]
        x72 = self.convs["72"](feature144, feature72, lap5)
        x36 = self.convs["36"](x72, feature36, lap4)
        x18 = self.convs["18"](x36, feature18, lap3)
        x9 = self.convs["9"](x18,[feature64], lap2)
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)), lap1)

        #consist-again
        x36 = self.convs["consist1"](x36, x72, lap4)
        x18 = self.convs["consist2"](x18, x36, lap3)
        x9 = self.convs["consist3"](x9, x18, lap2)
        x6 = self.convs["consist4"](x6, x9, lap1)
        
        outputs[("disp",0)] = self.sigmoid(self.convs["dispConvScale0"](x6) + self.High_Frequency_Upsample(self.convs["dispConvScale1"](x9)) + 0.1*lap1.mean(dim=1,keepdim=True))
        outputs[("disp",1)] = self.sigmoid(self.convs["dispConvScale1"](x9) + self.High_Frequency_Upsample(self.convs["dispConvScale2"](x18)) + 0.1*lap2.mean(dim=1,keepdim=True))
        outputs[("disp",2)] = self.sigmoid(self.convs["dispConvScale2"](x18) + self.High_Frequency_Upsample(self.convs["dispConvScale3"](x36)) + 0.1*lap3.mean(dim=1,keepdim=True))
        outputs[("disp",3)] = self.sigmoid(self.convs["dispConvScale3"](x36) + self.High_Frequency_Upsample(self.convs["dispConvScale4"](x72)) + 0.1*lap4.mean(dim=1,keepdim=True))
        

        outputs[("neww",0)] = img
        outputs[("neww",1)] = img2
        outputs[("neww",2)] = img3
        outputs[("neww",3)] = img4

        return outputs