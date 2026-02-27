from Image_Classification_From_Scratch.old_backend.activation_functions import *
from Image_Classification_From_Scratch.old_backend.linear import Linear

#Below are the libraries that will be used to perform tensor operations for normalisation, matrix-multiplication

# The pytorch library will be used to load the datasets, e.g. MNIST and CIFAR-10

import torch

from device import DEVICE
# Like the previous numpy version,
# for the shape of my input each row = new batch, and each column = neuron (feature)
# Sequential class manages the structure of the neural network.
# Instead of hard-coding layers, they are defined like this:
# model_features = [
#     ("linear", 784, 128),
#     ("activation","ReLU"),
#     ("linear", 128, 10),
# ]
class Sequential:
    device = DEVICE
    def __init__(self, model_features: list, device=DEVICE):
        # store the model configuration list for the neural network architecture
        self.model_features = model_features
        # store the device so that all tensors can be created on the same device
        # devices are (mps, cpu, cuda)
        self.device = device
        # holds the actual layer objects
        self.layers = []
        # this dictionary is used to map a.f. to their class implementations
        self.activations = {
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
            "Tanh": Tanh,
            "Softmax": Softmax,
        }
    # Loop through the model_features to instantiate layers
    def create_instances(self):
        for item in self.model_features:
            layer_type = item[0]
            # create a linear layer
            if layer_type == "linear":
                _, in_features, out_features = item
                new_layer = Linear(out_features, in_features, device=self.device)
                self.layers.append(new_layer)
            # set the activation function, technically it's not a layer, but it will still be stored inside layers
            elif layer_type == "activation":
                activation_name = item[1]
                self.layers.append(self.activations[activation_name]())
                print(f"Created Activation function: {activation_name}")

    def to(self, device):
        self.device = torch.device(device)

        for layer in self.layers:
            if hasattr(layer, "to"):
                layer.to(self.device)
        return self

    def forward(self, x):
        for layer in self.layers:

            x = layer.forward(x)

        return x