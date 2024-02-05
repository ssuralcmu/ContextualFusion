from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


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

    # def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
    #     return super().forward(torch.cat(inputs, dim=1))
    #     class ConvFuser(nn.Sequential):  
   

    def forward(self, inputs: List[torch.Tensor], night_mode) -> torch.Tensor:
        print("night_mode shape", night_mode.shape)
        print("input shape", inputs[0].shape)
        batch_size=(inputs[0].shape)[0]
        ones=np.ones((batch_size,1))
        night_mode=ones-0.67*night_mode
        print(night_mode)
        night_mode=np.reshape(night_mode,(batch_size,1,1,1))
        inputs[1] = inputs[1] * night_mode#(1 - night_time_offset) #Camera
        #print("night")

        #updated_inputs=
        return super().forward(torch.cat(inputs, dim=1))
