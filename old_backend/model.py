
from device import DEVICE
from Image_Classification_From_Scratch.old_backend.cross_entropy_loss import CrossEntropyLoss

class Model:
    def __init__(self, sequential, optimiser, lr=0.01 ):
        self.model_structure = sequential
        self.device = DEVICE
        self.optimiser = optimiser
        self.lr = lr
        self.cross_entropy = CrossEntropyLoss()
        self.y_true = None


    def train(self, y_true, input):
        self.y_true = y_true
        pred = self.model_structure.forward(input)
        # print(f"pred: {pred}")
        curr_loss = self.cross_entropy.forward(pred, self.y_true)
        # print(f"Current loss: {curr_loss}")
        grad = self.cross_entropy.backward()

        for layer in reversed(self.model_structure.layers):
            grad = layer.backward(grad)

            if hasattr(layer, "weights"):
                layer.weights = self.optimiser.step(grads=layer.grad_weights, input=layer.weights)
                layer.bias = self.optimiser.step(grads=layer.grad_biases, input=layer.bias)

        return curr_loss

