from cross_entropy_loss import *
from device import DEVICE
import torch



class ReLU:
    def __init__(self):
        self.input = None
        self.output = None
        self.grad_input = None
        self.grad_output = None

    def forward(self, input):
        zeros = torch.zeros(input.shape, device=DEVICE)
        self.input = input
        self.output = torch.maximum(input=self.input, other=zeros)
        return self.output

    def backward(self, grad_output):
        self.grad_input = grad_output
        self.grad_output = self.grad_input * (self.output > 0)
        return self.grad_output



class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None
        self.grad_output = None
        self.grad_input = None
        # self.zeros = torch.zeros(input.shape, device=DEVICE)

    def forward(self, input):
        self.input = input
        denominator = 1 + torch.exp(-self.input)
        self.output = 1 / denominator
        return self.output

    def backward(self, grad_output):
        self.grad_input = grad_output
        self.grad_output = self.grad_input * (self.output * (1 - self.output))
        return self.grad_output

class Tanh:
    def __init__(self):
        self.input = None
        self.output = None
        self.grad_input = None
        self.grad_output = None
        # self.zeros = torch.zeros(input.shape, device=DEVICE)

    def forward(self, input):
        self.input = input
        numerator = (torch.exp(self.input) - torch.exp(-self.input))
        denominator = (torch.exp(self.input) + torch.exp(-self.input))
        self.output = numerator / denominator
        return self.output

    def backward(self, grad_output):
        self.grad_input = grad_output
        self.grad_output = self.grad_input * (1 - torch.pow(self.output, 2))
        return self.grad_output

class Softmax:
    def __init__(self):
        self.input = None
        self.pred = None
        self.output = self.pred
        self.grad_input = None
        self.grad_output = None

    def forward(self, input):
        self.input = input

        max_val, _ = torch.max(input, dim=1, keepdim=True)
        shifted_input = input - max_val
        numerator = torch.exp(shifted_input)
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        self.pred = numerator / denominator


        return self.pred

    def backward(self, grad_output):
        self.grad_input = grad_output

        # Creating the Jacobian for the whole batch
        # Step A: Diagonal part (i = j)
        diag_part = torch.diag_embed(self.pred)

        # Step B: Outer product part (i != j)
        # p_reshaped becomes (3,4,1)
        p_reshaped = self.pred.unsqueeze(2)

        # bmm performs (3,4,1) x (3,1,4) -> (3,4,4)
        p_reshaped_transposed = p_reshaped.transpose(1, 2)  # swap dimensions 1 and 2
        outer_part = torch.bmm(p_reshaped, p_reshaped_transposed)

        softmax_derivative = diag_part - outer_part

        # self.grad_output = softmax_derivative * self.grad_input
        self.grad_output = torch.bmm(softmax_derivative, self.grad_input.unsqueeze(2)).squeeze(2)

        return self.grad_output