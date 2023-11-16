from typing import Any
import numpy as np

class Linear:
    """
    Linear: Y = dot(W, [X 1]^T)
    """
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # xavier initialization
        # better for sigmoid and tanh activation
        xavier_std = np.sqrt(1 / in_features)
        self.W = np.random.uniform(
            -xavier_std, xavier_std, (1 + in_features, out_features)
        )
        self.dW = np.zeros((1 + in_features, out_features))
    
    def __call__(self, X: np.ndarray):
        return self.forward(X)
    
    def forward(self, X: np.ndarray):
        """
        Argument:
            X (np.array): (batch size, ..., in_features)
        Return:
            Y (np.array): (batch size, ..., out_features)
        """
        # (batch size, ..., 1 + in_features)
        self.X_ = np.insert(X, 0, 1, axis = -1)
        Y = np.dot(self.X_, self.W)
        return Y

    def backward(self, delta: np.ndarray):
        """
        Argument:
            delta (np.array): (batch size, ..., out_features) dL/dY
        Return:
            out (np.array): (batch size, ..., in_features) dL/dX
        """
        self.dW = np.matmul(self.X_.T, delta)
        # (batch size, ..., 1 + in_features)
        self.dX_ = np.matmul(delta, self.W.T)
        self.dX = self.dX_[..., 1:]
        return self.dX
    