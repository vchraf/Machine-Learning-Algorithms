import numpy
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd

import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

"""### Plot"""

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    #axes.grid()

def set_figsize(figsize=(10, 2.5)):
    plt.figure.figsize = figsize

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[],
         xlim=None, ylim=None, xscale='linear', yscale='linear', axes=None):
  

    def isHas1axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))

    if isHas1axis(X): X = [X]
    if Y is None: Y = [[]*len(X)]
    if isHas1axis(Y):Y = [Y]
    if len(X) != len(Y): X = X * len(Y)


    plt.rcParams['figure.figsize'] = (7,5)
    if axes is None: 
      axes = plt.gca()#get get current axes
    axes.cla() #Clear the current active axes.
    for x, y in zip(X, Y):
      if len(x):axes.plot(x,y)
      else: axes.plot(y)
        #axes.plot(x,y) if len(x) else axes.plot(y)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def plotImages(imgs, r, c, titles=None, scale=1.5):
  figsize = (c * scale, r * scale)

  _, axes = plt.subplots(r, c, figsize=figsize)
  
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, imgs)):
    if torch.is_tensor(img):
      ax.imshow(img.numpy())
    else:
      # PIL Image 
      ax.imshow(img)
      ax.axes.get_xaxis().set_visible(False)
      ax.axes.get_yaxis().set_visible(False)
    if titles:
      ax.set_title(titles[i])
  return axes

"""### data

"""

def getFashionMNIST(batchSize, resize = None):
  ts = []
  if resize:
    ts.append(transforms.Resize(resize))
  ts.append(transforms.ToTensor())#transfrom the values of a tensor(C * H * W) from [0, 255] to [0, 1]
  tsCombo = transforms.Compose(ts) 
  mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform = tsCombo, download=True)
  mnist_test  = torchvision.datasets.FashionMNIST(root="../data", train=False, transform = tsCombo, download=True)
  X = data.DataLoader(mnist_train, batchSize, shuffle=True, num_workers= 2)
  y = data.DataLoader(mnist_test, batchSize, shuffle=False, num_workers= 2)
  return X, y

def get_labels(labels):
  if not hasattr(labels, "ndim") and not hasattr(labels, "__len__"): labels =  numpy.array([labels])
  text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt','sneaker', 'bag', 'ankle boot']
  return [text_labels[int(i)] for i in labels]

batch_size = 128
X,y = getFashionMNIST(batch_size)

def randomSimple(size):
  return next(iter(getFashionMNIST(size)[0]))

#X_, y_ = randomSimple(18)
#plotImages(X_.reshape(18, 28, 28), 3, 3, titles=get_labels(y_),scale=3)

"""### model"""

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



net = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    # Use three successive convolutional layers and a smaller convolution
                    # window. Except for the final convolutional layer, the number of output
                    # channels is further increased. Pooling layers are not used to reduce the
                    # height and width of input after the first two convolutional layers
                    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                    # Here, the number of outputs of the fully-connected layer is several
                    # times larger than that in LeNet. Use the dropout layer to mitigate
                    # overfitting
                    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                    # Output layer. Since we are using Fashion-MNIST, the number of classes is
                    # 10, instead of 1000 as in the paper
                    nn.Linear(4096, 10))

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

"""### Train and Test"""

def train(model, epochs, x):
  if isinstance(model,nn.Module):
    lossRec     = []
    epochsRec   = []
    accuracyRec = []
    for epoch in range(epochs):
      print("starEp_________",epoch)
      i_  = 0
      stp,correct       = 0,0
      for i, (img, label) in enumerate(x):
        img   = Variable(img.float())#.to(device)
        label = Variable(label)#.to(device)
        
        print("forw___",i_)
        #forwardPass
        out   = model(img)
        loss  = LossFun(out,label)
        
        print("back___",i_);i_+=1
        #BackwardPass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(out.data, 1) # for every row get the index of the class with the hight proba
        correct += (predicted == label).sum().item() #total number of the correct labels 
        stp+=label.size(-1)
      lossRec.append(loss.item())
      epochsRec.append(epoch)
      accuracyRec.append(correct/stp)
      print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' .format(epoch+1, epochs, stp, correct, loss.item(),accuracyRec[epoch]))
    return epochsRec, lossRec, accuracyRec

class AlexNett(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

from torchsummary import summary
model = AlexNet()
#X_test = torch.randn(1,1,224, 224)
#summary(model,input_size=X_test)

LossFun   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.05)

X,y = getFashionMNIST(batchSize=1,resize=224)
len(X),len(y)

output = train(model=AlexNet(),epochs= 10,x=X)

def accuracy(model,x):
  if isinstance(model,nn.Module):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in x:
            images = Variable(images.float()).to(device)
            labels = Variable(labels).to(device)
            outputs = model(images) # get the predections probabilitis
            _, predicted = torch.max(outputs.data, 1) # for every row get the index of the class with the hight proba
            total += labels.size(0) # total rows
            correct += (predicted == labels).sum().item() #total number of the correct labels 
        return (100 * correct / total)

print('Test Accuracy of the model on the 10000 test images: {} %'.format(accuracy(model,y)))

X_ = torch.randn(1, 1, 224, 224)
for layer in net:
  X_ = layer(X_)
  print(layer.__class__.__name__, 'output shape:\t', X_.shape)

X,y = getFashionMNIST(batchSize=batch_size)
ts = len(X)
ts,len(y)

from torchsummary import summary
model = MLP(784,10)
summary(model,(1,28,28))




















