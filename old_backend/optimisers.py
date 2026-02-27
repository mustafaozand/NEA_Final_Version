

class SGD:
    def __init__(self, lr=0.01):
        self.input = None
        self.output = None
        self.grads = None
        self.lr = lr

    def step(self, grads, input):
        self.input = input
        self.output = self.input - (self.lr * grads)
        return self.output

