import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
  def __init__(this, inSize, outSize):
      super(MLP,this).__init__()
      this.inSize   = inSize
      this.outSize  = outSize
      this.flatten  = nn.Flatten()
      this.layer0   = nn.Linear(this.inSize,this.inSize*2)
      this.af0      = nn.ReLU()

      this.layer1   = nn.Linear(this.inSize*2,this.inSize*4)
      this.af1      = nn.ReLU()

      this.layer2   = nn.Linear(this.inSize*4, this.outSize)
      this.af2      = nn.ReLU()

  def forward(this, x):
    out = this.flatten(x)

    out = this.layer0(out)
    out = this.af0(out)

    out = this.layer1(out)
    out = this.af1(out)

    out = this.layer2(out)
    out = this.af2(out)

    return out