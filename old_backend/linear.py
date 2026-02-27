from Image_Classification_From_Scratch.old_backend.cross_entropy_loss import *
from device import DEVICE


class Linear:
    def __init__(self, out_features, in_features, device=DEVICE):
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device(device)
        self.weights = torch.randn(out_features, in_features, device=self.device) * 0.01
        self.bias = torch.zeros(out_features, device=device)
        self.grad_input = None
        self.grad_weights = None
        self.grad_biases = None
        self.grad_output = None
        self.layer_outputs = None
        self.layer_inputs = None

    def to(self, device):
        self.device = torch.device(device)
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)

        return self

    def forward(self, layer_inputs):
        self.layer_inputs = layer_inputs
        self.layer_outputs = layer_inputs @ self.weights.T + self.bias
        return self.layer_outputs

    def backward(self, grad_output):
        # Connect the grad_output with the inputs
        self.grad_input = grad_output
        self.grad_weights = self.grad_input.T @ self.layer_inputs
        self.grad_biases = self.grad_input.sum(dim=0) # summation will be across each column
        self.grad_output = self.grad_input @ self.weights

        return self.grad_output
