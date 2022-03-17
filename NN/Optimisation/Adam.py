import numpy as np


class Adam():
    def __init__(this, learning_rate=0.001, b1=0.9, b2=0.999):
        this.learning_rate = learning_rate
        this.eps = 1e-8
        this.m = None
        this.v = None
        
        # Decay rates
        this.b1 = b1
        this.b2 = b2
    
    def update(this, w, grad_wrt_w):
        # If not initialized
        if this.m is None:
            this.m = np.zeros(np.shape(grad_wrt_w))
            this.v = np.zeros(np.shape(grad_wrt_w))
        
        this.m = this.b1 * this.m + (1 - this.b1) * grad_wrt_w
        this.v = this.b2 * this.v + (1 - this.b2) * np.power(grad_wrt_w, 2)

        m_hat = this.m / (1 - this.b1)
        v_hat = this.v / (1 - this.b2)

        this.w_updt = this.learning_rate * m_hat / (np.sqrt(v_hat) + this.eps)

        return w - this.w_updt