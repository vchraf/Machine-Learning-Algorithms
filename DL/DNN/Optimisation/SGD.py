import numpy as np

class StochasticGradientDescent():
    def __init__(this, learning_rate=0.01, momentum=0):
        this.learning_rate = learning_rate 
        this.momentum = momentum
        this.w_updt = None

    def update(this, w, grad_wrt_w):
        # If not initialized
        if this.w_updt is None:
            this.w_updt = np.zeros(np.shape(w))
        # Use momentum if set
        this.w_updt = this.momentum * this.w_updt + (1 - this.momentum) * grad_wrt_w
        # Move against the gradient to minimize loss
        return w - this.learning_rate * this.w_updt