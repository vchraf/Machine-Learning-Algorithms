import numpy as np

class Loss(object):
  def loss(this, y_true, y_pred): return NotImplementedError()
  def gradient(this, y, y_pred): raise NotImplementedError()
  def acc(this, y, y_pred): return 0