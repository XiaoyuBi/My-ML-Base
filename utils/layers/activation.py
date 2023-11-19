import numpy as np

class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,

    Note that these activation functions are scalar operations. I.e, they
    shouldn't change the shape of the input.
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def backward(self):
        raise NotImplemented

class Sigmoid(Activation):
    """
    Sigmoid: logits = 1 / (1 + exp(-x))
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1.0 / (1 + np.exp(-x))
        return self.state

    def backward(self):
        return self.state * (1 - self.state)

class Tanh(Activation):
    """
    Tanh: logits = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.state

    def backward(self):
        return 1.0 - self.state * self.state

class ReLU(Activation):
    """
    ReLU non-linearity
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.where(x > 0.0, x, 0.0)
        return self.state

    def backward(self):
        return np.where(self.state > 0.0, 1.0, self.state)
