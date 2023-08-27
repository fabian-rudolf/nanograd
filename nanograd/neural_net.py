import random
from nanograd.value import Value

class Module:
    def zero_gradient(self):
        for parameter in self.parameters():
            parameter.gradient = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, is_nonlinear=True):
        self.weight = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(0)
        self.is_nonlinear = is_nonlinear

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.weight, x)), self.bias)
        return act.relu() if self.is_nonlinear else act

    def parameters(self):
        return self.weight + [self.bias]

    def __repr__(self):
        return f"{'ReLU' if self.is_nonlinear else 'Linear'} neuron({len(self.weight)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    MLP: Multi-Layer perceptron
    """
    def __init__(self, number_of_inputs, number_of_outputs):
        sz = [number_of_inputs] + number_of_outputs
        self.layers = [
            Layer(
                sz[i], 
                sz[i+1], 
                is_nonlinear=i!=len(number_of_outputs)-1) for i in range(len(number_of_outputs))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
