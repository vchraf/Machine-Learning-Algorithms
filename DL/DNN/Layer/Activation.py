import numpy as np
from .Layer import Layer
from .ActivationFun import ReLU, Sigmoid
class Activation(Layer):  
  def __init__(this, name):
      activation_functions = {'relu': ReLU, 'sigmoid': Sigmoid}
      this.activation_name = name
      this.activation_func = activation_functions[name]()
      this.trainable = True

  def layerName(this):
      return "Activation (%s)" % (this.activation_func.__class__.__name__)

  def forwardPass(this, X, training=True):
      this.layer_input = X
      return this.activation_func(X)

  def backwardPass(this, accumGrad):
      return accumGrad * this.activation_func.grad(this.layer_input)

  def outputShape(this):
      return this.inputShape