import torch

class CrossEntropyLoss:
    def __init__(self):
        self.y_true = None
        self.pred = None
        self.avg_loss = None
        self.epsilon = 1e-9
        self.batch_size = None
        self.grad_output = None

    # since our model will be trained in batches, the loss will the total loss over each sample averaged over the number of samples there are in a batch
    def forward(self, predictions, y_true):
        self.pred = predictions
        self.y_true = y_true
        self.batch_size = self.pred.shape[0] # (batch_size, neurons)
        batch_loss = -torch.sum(self.y_true * torch.log(predictions + self.epsilon))
        self.avg_loss = batch_loss / self.batch_size

        return self.avg_loss

    def backward(self):
        cross_entropy_derivative = -(self.y_true / (self.pred + self.epsilon))

        # We will divide by batch_size to average out across the batch, this way we can keep th gradient scale consistent.
        self.grad_output = (cross_entropy_derivative / self.batch_size)

        return self.grad_output
