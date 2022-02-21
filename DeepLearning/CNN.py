from torch.nn.modules import padding
class CNN(nn.Module):
  def __init__(this,nbr_classes):
      super(CNN, this).__init__()

      this.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
      this.bn1  = nn.BatchNorm2d(16)
      this.af1  = nn.ReLU()
      nn.init.xavier_uniform_(this.cnn1.weight)
      this.avgP1=nn.AvgPool2d(kernel_size=2)

      this.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
      this.bn2  = nn.BatchNorm2d(32)
      this.af2  = nn.ReLU()
      nn.init.xavier_uniform_(this.cnn2.weight)
      this.avgP2=nn.AvgPool2d(kernel_size=2)

      this.fc1  = nn.Linear(32*7*7,nbr_classes) #Fully Connect layers
    
  def forward(this, x):
    out = this.cnn1(x)
    out = this.bn1(out)
    out = this.af1(out)
    out = this.avgP1(out)
    
    out = this.cnn2(out)
    out = this.bn2(out)
    out = this.af2(out)
    out = this.avgP2(out)


    out = out.view(out.size(0),-1)
    out = this.fc1(out)

    return out