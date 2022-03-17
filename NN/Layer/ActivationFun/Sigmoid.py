import numpy as np

class Sigmoid():
  def __call__(this, x): 
      return 1 / (1 + np.exp(-x))
  
  def grad(this, x): 
      return this.__call__(x) * (1 - this.__call__(x))