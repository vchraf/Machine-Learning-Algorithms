class Layer(object):
  def setInputShape(this, shape): this.inputShape = shape
  def layerName(this): return this.__class__.__name__
  def parameters(this): return 0
  def forwardPass(this, X, training): raise NotImplementedError()
  def backwardPass(this, accumGrad): raise NotImplementedError()
  def outputShape(this): raise NotImplementedError()