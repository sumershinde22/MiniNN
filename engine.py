import math

class Value:

    # Initialize the Value object with data and optional parameters for operation tracking and labels
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # Provide a simple string representation for the Value object
    def __repr__(self):
        return f"Value(data={self.data})"

    # Defines addition for Value objects or a Value and a number
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    # Defines multiplication for Value objects or a Value and a number
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    # Defines power operation for Value objects, supporting only integer or float powers
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    # Enables multiplication when Value is on the right-hand side
    def __rmul__(self, other):
        return self * other

    # Defines true division for Value objects
    def __truediv__(self, other):
        return self * other ** -1

    # Defines negation for Value objects
    def __neg__(self):
        return self * -1

    # Defines subtraction for Value objects
    def __sub__(self, other):
        return self + (-other)

    # Handles addition when Value is on the right-hand side
    def __radd__(self, other):
        return self + other

    # Implements the hyperbolic tangent function and its gradient
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    # Implements the exponential function and its gradient
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    # Implements backpropagation for computational graphs
    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
