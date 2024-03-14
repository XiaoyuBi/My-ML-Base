import copy
import torch
from torchvision.datasets import MNIST

class MyMNIST(MNIST):
    """
    The base of this class is taken from GitHub Gist, then I adapted it to my needs.
    link: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.float().div(255)
        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # labels 0-9, not used in this project
        self.labels = self.targets

        # Since I'm working with autoencoders, 'targets' becomes a copy of the data. 
        # The labels are now stored in the variable 'labels'. 
        self.targets = copy.deepcopy(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target, label = self.data[index], self.targets[index], self.labels[index]
        return data, target, label


class NoisyMNIST(MyMNIST):
    """
    Data Augmented MNIST dataset with random noises
    """

    def __init__(self, *args, sigma = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data += sigma * torch.randn(self.data.shape)
        self.data = torch.clamp(self.data, 0., 1.)
