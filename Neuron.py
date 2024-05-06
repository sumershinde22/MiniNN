import random
from engine import Value


class Neuron:

    # Initialize a Neuron with random weights and bias
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    # Callable method for neuron activation using weighted inputs and bias
    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    # Returns all parameters of the neuron (weights and bias)
    def parameters(self):
        return self.w + [self.b]


class Layer:

    # Initialize a Layer containing multiple neurons
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    # Callable method to process input through the layer of neurons
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    # Aggregate all parameters from all neurons in the layer
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

    # Initialize a Multi-Layer Perceptron (MLP) with specified sizes
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    # Callable method to process input through all layers of the MLP
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # Aggregate all parameters from all layers in the MLP
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
