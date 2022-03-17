import numpy as np
from .Loss import Loss

class SquareLoss(Loss):
  def __init__(this): 
      pass
  
  def loss(this, y, y_pred):
      return  0.5 * np.power((y - y_pred), 2)
  
  def grad(this, y, y_pred):
      return -(y - y_pred)
  
  def acc(this, y, y_pred):
      return np.sum(y == y_pred, axis=0) / len(y)