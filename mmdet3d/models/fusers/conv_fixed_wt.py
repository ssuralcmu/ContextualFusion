from typing import List
import numpy as np
import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        #print(in_channels,out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    # def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
    #     return super().forward(torch.cat(inputs, dim=1))
    #     class ConvFuser(nn.Sequential):  
   

    def forward(self, inputs: List[torch.Tensor], context) -> torch.Tensor:
        #print("night_mode shape", night_mode.shape)
        #print("input shape", inputs[0].shape)
        batch_size=(inputs[0].shape)[0]
        ones=np.ones((batch_size,1))
        #print("init_night_mode=",night_mode)
        #night_mode=night_mode.astype('float')
        array_context=np.zeros((batch_size,3))
        for i in range(batch_size):
            for j in range(3):      
                array_context[i][j]=int(context[i][0][j])

        
        #night_mode=torch.from_numpy(np.reshape(array_context,(batch_size,3))).to(inputs[1].get_device(),dtype=torch.float)
        #gate_out = self.gating(night_mode)
        #gate_out = torch.unsqueeze(torch.unsqueeze(gate_out, -1), -1)
        print(array_context[:,0],array_context[:,1])
        #if both rain and night, night dominates, else whichever
        night_mode=ones-0.19*array_context[:,1]#ones-0.35*array_context[:,0]#-0.19*array_context[:,1]*(1-array_context[:,0])
        #print(night_mode,ones)
        print("final night_mode=",night_mode)
        night_mode=torch.from_numpy(np.reshape(night_mode,(batch_size,1,1,1))).to(inputs[1].get_device(),dtype=torch.float)
        inputs[1] = inputs[1] * night_mode#(1 - night_time_offset) #Camera
        #print("night")

        #updated_inputs=
        return super().forward(torch.cat(inputs, dim=1))
