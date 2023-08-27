
# nanograd

This repository is based on and inspired by `micrograd` by karpathy.

* define any directed acyclic graph of scalar values
* chain values by available mathematical operations
* define neurons by chaining individual adding and multiplication steps
* rules derived from linear algebra for calculating derivatives apply to the backpropagation steps through the network
* the gradient of each value dimension estimates how the result function's value can be tweaked into the desired direction during model training
* integrates with PyTorch
* define deep neural nets by their representation as a directed acyclic graph of math operations applied to input values yielding an output value

### Installation

local
```bash
pip install nanograd --user -e .
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from nanograd.value import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.of:.4f}') # prints 24.7041, the outcome of this forward pass
g.backpropagate()
print(f'{a.gradient:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.gradient:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `nanograd.neural_net` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from nanograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT
