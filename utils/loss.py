import numpy as np

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """
    def __init__(self):
        self.y_ = None # prediction
        self.y = None # true labels
        self.loss = None

    def __call__(self, y_, y):
        return self.forward(y_, y)

    def forward(self, y_, y):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class MSELoss(Criterion):
    """
    MSE Loss = (y_ - y) ^ 2
    """
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, y_: np.ndarray, y: np.ndarray):
        """
        Argument:
            y_ (np.array): (batch size, ...)
            y (np.array): (batch size, ...)
        Return:
            out (np.array): ()
        """
        assert y_.shape == y.shape, f"Inputs shape not matching! {y_.shape}, {y.shape}"
        self.y_ = y_
        self.y = y
        self.total_size = np.prod(y.shape)
        # use default "mean" reduction
        self.loss = np.sum((y_ - y) * (y_ - y)) / self.total_size
        return self.loss 
    
    def backward(self):
        """
        Return:
            out (np.array): (batch size, ...)
        """
        return 2 * (self.y_ - self.y) / self.total_size
    