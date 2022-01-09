from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch import Tensor

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class LearnedReLU(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int
    def __init__(self, num_parameters: int = 1, init: float = 1.0) -> None:
        self.num_parameters = num_parameters
        super(LearnedReLU, self).__init__()
        self.a = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input: Tensor) -> Tensor:
        # input: n x N x T -> n x T x N
        return self.a*input.clamp(min=0.0)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)

class ShiftedSigmoid(nn.Module):
    def __init__(self):
        super(ShiftedSigmoid, self).__init__()

    def forward(self, x):
        return 1.2 / (1 + torch.exp(-(1 / 1.6) * x))

class LearnedSigmoid(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int
    def __init__(self, num_parameters: int = 1, a: float = 1.0, b: float = 1.0) -> None:
        self.num_parameters = num_parameters
        super(LearnedSigmoid, self).__init__()  
        self.a = Parameter(torch.Tensor(num_parameters).fill_(b))
        self.b = Parameter(torch.Tensor(num_parameters).fill_(b))

    def forward(self, input: Tensor) -> Tensor:
        # input: n x N x T -> n x T x N
        return (self.b / (1 + torch.exp( -self.a*input.transpose(-1,-2) ))).transpose(-1,-2)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)

class Learnable_sigmoid(nn.Module):
    def __init__(self, in_features=257):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True  # set requiresGrad to true!

        # self.scale = nn.Parameter(torch.ones(1))
        # self.scale.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        return 1.2 / (1 + torch.exp(-(self.slope) * x.transpose(-1,-2))).transpose(-1,-2)

def linear():
    return nn.Identity()


def relu():
    return nn.ReLU()


def prelu():
    return nn.PReLU()


def leaky_relu():
    return nn.LeakyReLU()

def sigmoid():
    return nn.Sigmoid()

def shifted_sigmoid():
    return ShiftedSigmoid()

def learned_sigmoid():
    return LearnedSigmoid()

def learned_relu():
    return LearnedReLU()

def softmax(dim=None):
    return nn.Softmax(dim=dim)

def tanh():
    return nn.Tanh()

def gelu():
    return nn.GELU()

def geglu():
    return GEGLU()

def swish():
    return Swish()


def register_activation(custom_act):
    """Register a custom activation, gettable with `activation.get`.
    Args:
        custom_act: Custom activation function to register.
    """
    if custom_act.__name__ in globals().keys() or custom_act.__name__.lower() in globals().keys():
        raise ValueError(f"Activation {custom_act.__name__} already exists. Choose another name.")
    globals().update({custom_act.__name__: custom_act})


def get(identifier):
    """Returns an activation function from a string. Returns its input if it
    is callable (already an activation for example).
    Args:
        identifier (str or Callable or None): the activation identifier.
    Returns:
        :class:`nn.Module` or None
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError("Could not interpret activation identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret activation identifier: " + str(identifier))

"""
x = torch.Tensor([-5, -4 , -3, 1, 2, 3])
print(x)
act = LearnedReLU()
y = act(x)
print(y)
"""