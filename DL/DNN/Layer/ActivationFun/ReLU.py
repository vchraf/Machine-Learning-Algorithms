import numpy as np

class ReLU():
  def __call__(this, x):
      return np.where(x >= 0, x, 0)

  def grad(this, x):
      return np.where(x >= 0, 1, 0)
      