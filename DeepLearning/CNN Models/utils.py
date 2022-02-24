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
    if Y is None: X, Y = [[]] * len(X), X
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





















