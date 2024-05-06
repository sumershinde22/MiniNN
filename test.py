from Neuron import MLP
from GradientDescent import Gradient
from graph import draw_dot

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
grad = Gradient(n, xs, ys)
grad.calculate()
print(draw_dot(n(x)))