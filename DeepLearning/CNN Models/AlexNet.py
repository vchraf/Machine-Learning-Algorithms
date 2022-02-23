import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

class AlexNet(nn.Module):
  def __init__(this):
      super(AlexNet, this).__init__()  
      this.features = nn.Sequential(
        #Convolutional layer 1
          nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4), # Output Size 96 x 55 x 55
          nn.MaxPool2d(kernel_size=3, stride= 2),# Output Size 96 x 27 x 27
          nn.ReLU(),
        
        #Convolutional layer 2
          nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), # Output Size 256 x 26 x 26
          nn.MaxPool2d(kernel_size=3, stride= 2), # Output Size 256 x 13 x 13
          nn.ReLU(),

        #Convolutional layer 3
          nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), nn.ReLU(), # Output Size 384 x 11 x 11
          nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), nn.ReLU(), # Output Size 384 x 9 x 9
          nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), # Output Size 256 x 7 x 7
          nn.MaxPool2d(kernel_size=3, stride= 2),  # Output Size 256 x 5 x 5
          nn.Flatten()  #transform from (256, 5, 5) -> (4600, 1)
        )
      
      
      this.classifier   = nn.Sequential(
          nn.Linear(in_features=6400, out_features=4096), nn.ReLU(),nn.Dropout(p=0.5),
          nn.Linear(in_features=4096, out_features=4096), nn.ReLU(),nn.Dropout(p=0.5),
          nn.Linear(in_features=4096, out_features=10)
      )
    
  def forward(this, x):
    x = this.features(x)
    x = this.classifier(x)
    return x