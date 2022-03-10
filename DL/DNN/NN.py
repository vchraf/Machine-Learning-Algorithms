from email import utils
import numpy as np
from ...utils.utils import batch_iterator


class NN():
  def __init__(this, optimizer, loss, validationData = None):
    this.optimizer  = optimizer
    this.layers     = []
    this.errors     = {"training": [], "validation":[]}
    this.lossFun    = loss()
    this.valSet     = None
    
    if validationData:
      X, y = validationData
      this.valSet = {"X": X, "y": y}

  def setTrainable(this, trainable):
    for layer in this.layers:
      layer.trainable = trainable

  def add(this, layer):
    if this.layers:
      layer.setInputShape(shape = this.layers[-1].outputShape())
    
    if hasattr(layer, 'initialize'):
      layer.initialize(optimizer=this.optimizer)

    this.layers.append(layer)
    
  def _forwardPass(this, X, training=True):
      layer_output = X
      
      for ix, layer in enumerate(this.layers):
          layer_output = layer.forwardPass(layer_output, training)

      return layer_output

  def _backwardPass(this, loss_grad):
      for layer in reversed(this.layers):
          loss_grad = layer.backwardPass(loss_grad)
    
  def test_OnBatch(this, X, y):
    y_pred  = this._forwardPass(X, training = False)
    loss    = np.mean(this.lossFun.loss(y, y_pred))
    acc     = this.lossFun.acc(y, y_pred)
    return loss, acc

  def train_OnBatch(this, X, y):
    y_pred    = this._forwardPass(X)
    loss      = np.mean(this.lossFun.loss(y= y, y_pred= y_pred))
    acc       = this.lossFun.acc(y, y_pred)
    loss_grad = this.lossFun.grad(y, y_pred)
    this._backwardPass(loss_grad = loss_grad)
    return loss, acc
  
  def fit(this, X, y, n_epochs, batch_size):
    for _  in range(n_epochs):
        batch_error = []
        for X_batch, y_batch in batch_iterator(X,y,batch_size):
          loss,acc= this.train_OnBatch(X_batch, y_batch)

          batch_error.append(loss)

        this.errors["training"].append(np.mean(batch_error))
        if this.valSet is not None:
          val_loss=this.test_OnBatch(this.valSet["X"], this.valSet["y"]) 
          this.errors["validation"].append(val_loss)
    print("TrainingDone.")
    return this.errors["training"], this.errors["validation"]
  def predict(this, X):
    return this._forwardPass(X, training=False)