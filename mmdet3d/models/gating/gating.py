
from typing import List
import numpy as np
import torch
from torch import nn

from mmdet3d.models.builder import GATING

__all__ = ["Gating"]

@GATING.register_module()
class Gating(nn.Module):
    def __init__(self):
        super().__init__()
        self.gating=nn.Linear(1, 336, bias = False)
        self.weights=torch.ones((336,1))
        self.weights[0:80]*=0 #0 for night and rain
        self.weights[80:336]*=0 #0.7 for night an 0 for rain
        self.gating.weight=torch.nn.Parameter(self.weights)
        self.count=0

    def forward(self, inputs: List[torch.Tensor], context) -> torch.Tensor:
        batch_size=(inputs[0].shape)[0]
        ones=np.ones((batch_size,1))
        array_context=np.zeros((batch_size,3))
        for i in range(batch_size):
            for j in range(3):      
                array_context[i][j]=int(context[i][0][j]) 
                #print("context:",array_context[i][j])
        night_mode=torch.from_numpy(np.reshape(array_context,(batch_size,3))).to(inputs[1].get_device(),dtype=torch.float)
        #print(night_mode)
        #print(night_mode[0][0])
        #print(night_mode[0][1])
        gate_in = torch.reshape(night_mode[0][1],(1,1))#For rain
        #gate_in = torch.reshape(night_mode[0][0],(1,1))#For night mode
        # dev=gate_in.device
        # mod=Gating().to(dev)
        gate_out = self.gating(gate_in)
        gate_out = torch.unsqueeze(torch.unsqueeze(gate_out, -1), -1)
        self.count+=1
        if self.count%10==0:
            if(gate_in==1):
                print("night",gate_in)
                print("Sample LID weight: ",1-gate_out[:,2])    
                print("Sample CAM weight: ",1-gate_out[:,100])
        if self.count%100==0:
                print("Sample LID weight: ",1-gate_out[:,2])    
                print("Sample CAM weight: ",1-gate_out[:,100])
        inputs[0] = inputs[0] * (1-gate_out[:,0:80])
        inputs[1] = inputs[1] * (1-gate_out[:,80:336])
        return inputs
