#This is the conv function for rain/no rain + day/night + workzone/no workzone

from typing import List
import numpy as np
import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.count=0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gating = nn.Sequential(
            nn.Linear(3, 1, bias = False),
            nn.Sigmoid()
        )
        self.conv=nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.rl=nn.ReLU(True)

        for i_ in self.gating:
            if isinstance(i_, nn.Linear):
                torch.nn.init.constant_(i_.weight, 1.735) 
   

    def forward(self, inputs: List[torch.Tensor], context) -> torch.Tensor:
        batch_size=(inputs[0].shape)[0]
        ones=np.ones((batch_size,1))
        #print("Night mode print")
        #print(context)
        #print(context.shape)
        array_context=np.zeros((batch_size,3))
        for i in range(batch_size):
            for j in range(3):      
                array_context[i][j]=int(context[i][0][j])

        
        night_mode=torch.from_numpy(np.reshape(array_context,(batch_size,3))).to(inputs[1].get_device(),dtype=torch.float)
        gate_out = self.gating(night_mode)
        gate_out = torch.unsqueeze(torch.unsqueeze(gate_out, -1), -1)
        #print(gate_out)
        #print(gate_out.shape)
        self.count+=1
        if self.count%100==0:
        	print("weight_cam_to_lid: ",2*(1-gate_out))     
        inputs[1] = inputs[1] * 2 * (1-gate_out)
        #inputs[1] = inputs[1] * 2 * (1-gate_out)
        x1=torch.cat(inputs, dim=1)
        x1=self.conv(x1)
        x1=self.bn(x1)
        out=self.rl(x1)
        return out
        
        
        
        
        
