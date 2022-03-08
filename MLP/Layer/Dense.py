import math
import copy
import numpy as np
from .Layer import Layer

class Dense(Layer):
  def __init__(this, nbrUnits, inputShape = None):
    this.layerInput = None
    this.inputShape = inputShape
    this.nbrUnits   = nbrUnits
    this.trainable  = True
    this.w          = None # Weight
    this.w0         = None # Bias

  #Weight initialization
  def initialize(this, optimizer):
    limit = 1 / math.sqrt(this.inputShape[0])
    this.w  = np.random.uniform(-limit, limit, (this.inputShape[0], this.nbrUnits))
    this.w0 = np.zeros((1, this.nbrUnits))

    this.w_opt  = copy.copy(optimizer)
    this.w0_opt = copy.copy(optimizer)
  
  #get the number of parameters
  def parameters(this): 
    return np.prod(this.w.shape) + np.prod(this.w0.shape)

  #forward pass X.w + b
  def forwardPass(this, X, training=True):
    this.layerInput = X
    return X.dot(this.w) + this.w0
  
  def backwardPass(this, accumGrad):
    # Save weights used during forwards pass
    w = this.w

    if this.trainable:
        # Calculate gradient w.r.t layer weights
        grad_w = this.layerInput.T.dot(accumGrad)
        grad_w0 = np.sum(accumGrad, axis=0, keepdims=True)

        # Update the layer weights
        this.w  = this.w_opt.update(this.w, grad_w)
        this.w0 = this.w0_opt.update(this.w0, grad_w0)

    # Return accumulated gradient for next layer
    # Calculated based on the weights used during the forward pass
    accum_grad = accumGrad.dot(w.T)
    return accum_grad

  def outputShape(this):
    return (this.nbrUnits, ) 