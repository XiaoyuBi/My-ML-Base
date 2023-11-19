import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from utils.layers.linear import Linear
from utils.loss import MSELoss

class LinearRegression:
    """
    Linear Regression training and scoring Interface
    """
    
    def __init__(self, lr: float = 0.1, momentum: float = 0.0, 
                 n_iter: int = 100):
        self.lr = lr
        self.momentum = momentum
        self.n_iter = n_iter
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Argument:
            X (np.array): (batch size, in_features)
            y (np.array): (batch size, 1)
        """
        in_features = X.shape[1]
        self.linear = Linear(in_features, 1)
        self.mseloss = MSELoss()
        self.training_losses = []

        for _ in range(self.n_iter):
            # forward
            y_ = self.linear(X)
            loss = self.mseloss(y_, y)
            self.training_losses.append(loss)
            # backward
            dL_dy_ = self.mseloss.backward()
            dL_dX = self.linear.backward(dL_dy_)
            # update
            self.linear.step(self.lr, self.momentum)
            self.linear.zero_grad()
    
    def forward(self, X: np.ndarray):
        if not hasattr(self, "linear"):
            raise ValueError("Train the linear regression model first!")
        return self.linear(X)
