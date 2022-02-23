import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable


class LeNet(nn.Module):
  
  def __init__(this):
      super(LeNet,this).__init__()
      this.conv_0 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
      this.avgP_0 = nn.AvgPool2d(kernel_size=2, stride= 2)
      this.relu_0 = nn.ReLU()

      this.conv_1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
      this.avgP_1 = nn.AvgPool2d(kernel_size=2, stride= 2)
      this.relu_1 = nn.ReLU()
    
    
      this.flat   = nn.Flatten()
      this.fc_0   = nn.Linear(16*5*5, 120)
      this.relu_2 = nn.ReLU()
   
      this.fc_1   = nn.Linear(120, 84)
      this.relu_3 = nn.ReLU()
   
      this.fc_2   = nn.Linear(84, 10)
      this.relu_4   = nn.ReLU()
  
  def forward(this, x):
    out = this.relu_0(this.avgP_0(this.conv_0(x)))     
    out = this.relu_1(this.avgP_1(this.conv_1(out)))
    out = this.relu_2(this.fc_0(this.flat(out)))
    out = this.relu_3(this.fc_1(out))
    out = this.fc_2(out)
    out = this.relu_4(out)

    return out