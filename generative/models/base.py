from torch import nn 
from abc import abstractmethod

from models.types_ import *

class BaseGenerativeModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(BaseGenerativeModel, self).__init__(*args, **kwargs)
    
    def sample(self, *args, **kwargs) -> Tensor:
        """
        Sample out-of-sample results from trained model
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *args, **kwargs) -> Dict[str, Tensor]:
        pass
    