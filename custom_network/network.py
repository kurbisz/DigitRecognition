import random
from typing import Callable, Optional

import numpy as np


class Network(object):

    def __init__(
        self,
        sizes: list[int],
        activation: Callable,
        activation_derivative: Callable,
        eval_func: Callable,
    ) -> "Network":
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [
            np.random.randn(y, x) * np.sqrt(2 / x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.eval_func = eval_func

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        for b, w in zip(self.biases, self.weights):
            a = self.activation(np.dot(w, a) + b)
        return a

    def start(
        self,
        training_data: list[tuple],
        epochs: int,
        mini_batch_size: int,
        eta: float,
        test_data: Optional[list] = None,
    ) -> list[float]:
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        epochsVal = []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:(k + mini_batch_size)]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_simple_batch(mini_batch, eta)

            ev = self.evaluate(test_data)
            print(f"Epoch {(j+1)}: {ev/n_test*100}")
            epochsVal.append(ev / n_test * 100)
        return epochsVal

    def update_simple_batch(self, mini_batch: list[tuple], eta: float) -> None:
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            prob_b, prob_w = self.back_propagation(x, y)
            delta_b = [nb + dnb for nb, dnb in zip(delta_b, prob_b)]
            delta_w = [nw + dnw for nw, dnw in zip(delta_w, prob_w)]

        self.weights = [
            w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, delta_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, delta_b)
        ]

    def back_propagation(
        self,
        x: list,
        y: list,
    ) -> tuple[list[float], list[float]]:
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * self.activation_derivative(zs[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = self.activation_derivative(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            delta_b[-layer] = delta
            delta_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return (delta_b, delta_w)

    def evaluate(self, test_data: list[tuple]) -> int:
        test_results = [
            (self.eval_func(self.feedforward(x)), y) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(
        self,
        output_activations: list[float],
        y: list[float],
    ) -> list[float]:
        return output_activations - y
