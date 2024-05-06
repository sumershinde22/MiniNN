from engine import Value


class Gradient:

    # Initialize the Gradient with model, inputs, and target outputs
    def __init__(self, n, xs, ys):
        self.n = n
        self.xs = xs
        self.ys = ys

    def calculate(self):
        loss = Value(1.0)
        k = 0
        ypred = 0

        # Calculate the gradient descent to minimize the loss function
        # while loss.data > 0.0001:
        while k< 20:
            k += 1
            # forward pass
            ypred = [self.n(x) for x in self.xs]
            loss = sum((yout - ygt) ** 2 for ygt, yout in zip(self.ys, ypred))

            # backward pass
            for p in self.n.parameters():
                p.grad = 0.0
            loss.backward()

            # update
            for p in self.n.parameters():
                p.data += -0.1 * p.grad

            print(k, loss.data)
        for i in ypred:
            print(i.data)
        print(self.n.parameters())