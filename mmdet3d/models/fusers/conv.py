#This is the conv function for rain/no rain + day/night + workzone/no workzone

from typing import List
import numpy as np
import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]

# @FUSERS.register_module()
# class Gating(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.gating=nn.Linear(1, 336, bias = False)
#         self.weights=torch.ones((336,1))
#         self.weights[0:80]*=0
#         self.weights[80:336]*=0
#         self.gating.weight=torch.nn.Parameter(self.weights)


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        print(in_channels,out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


# @FUSERS.register_module()
# class ConvFuser(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int) -> None:
#         super().__init__()
#         self.count=0
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         #self.gating = nn.Sequential(
#         #    nn.Linear(1, 336, bias = False),
#         #    nn.Sigmoid()
#         #)
#         self.gating=nn.Linear(1, 336, bias = False)
#         self.conv=nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False)
#         self.bn=nn.BatchNorm2d(out_channels)
#         self.rl=nn.ReLU(True)
#         #print("dim",self.gating.weight.shape,flush=True)
#         weights=torch.ones((336,1))
#         weights[0:80]*=0
#         weights[80:336]*=0
#         self.gating.weight=torch.nn.Parameter(weights)
#         #count=0
#         #for i_ in self.gating:
#         #    if isinstance(i_, nn.Linear):
#         #        torch.nn.init.constant_(i_.weight, 1.735/336) 
        
#     def forward(self, inputs: List[torch.Tensor], context) -> torch.Tensor:
        # batch_size=(inputs[0].shape)[0]
        # ones=np.ones((batch_size,1))
        # array_context=np.zeros((batch_size,3))
        # for i in range(batch_size):
        #     for j in range(3):      
        #         array_context[i][j]=int(context[i][0][j])
        # night_mode=torch.from_numpy(np.reshape(array_context,(batch_size,3))).to(inputs[1].get_device(),dtype=torch.float)
        # gate_in = torch.reshape(night_mode[0][0],(1,1))
        # gate_out = self.gating(gate_in)
        # gate_out = torch.unsqueeze(torch.unsqueeze(gate_out, -1), -1)
        # #print(gate_out)
        # #print(gate_out.shape)
        # self.count+=1
        # if self.count%50==0:
        #     print("Sample LID weight: ",1-gate_out[:,2])    
        #     print("Sample CAM weight: ",1-gate_out[:,100])
        # #print(inputs[0].shape,gate_out.shape) #80
        # #print(inputs[1].shape,gate_out.shape) #256
        # #print(inputs[0][0][0][0])
        # #print(inputs[0].shape)
        # #print(inputs[1].shape)
        # inputs[0] = inputs[0] * (1-gate_out[:,0:80])
        # inputs[1] = inputs[1] * (1-gate_out[:,80:336])
        # #print(inputs[0][0][0][0])
        # #print("Input shape: ",inputs[0].shape,inputs[1].shape)
        # #inputs[1] = inputs[1] * 2 * (1-gate_out)
        # x1=torch.cat(inputs, dim=1)
        # x1=self.conv(x1)
        # x1=self.bn(x1)
        # out=self.rl(x1)
        # return out
        
        
        
        
        
