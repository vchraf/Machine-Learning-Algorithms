import numpy as np

def oneHot(x, nbrCol=None):
  if not nbrCol: nbrCol = np.max(x) + 1
  one_hot = np.zeros((x.shape[0], nbrCol))
  one_hot[np.arange(x.shape[0]), x] = 1
  return one_hot

def batch_iterator(X, y=None, batch_size=64):
  n_samples = X.shape[0]
  for i in np.arange(0, n_samples, batch_size):
    begin, end = i, min(i+batch_size, n_samples)
    if y is not None:
      yield X[begin:end], y[begin:end]
    else:
      yield X[begin:end]
