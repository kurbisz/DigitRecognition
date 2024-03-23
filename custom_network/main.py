import math
import random

import numpy as np
from network import Network


def sigmoid(z: np.ndarray) -> np.ndarray:
    r = 1.0 / (1.0 + np.exp(-z))
    return r


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))


def ReLU(z: np.ndarray) -> np.ndarray:
    t = z.flatten()
    return np.array([[max(0.0, a)] for a in t])


def ReLU_derivative(z: np.ndarray) -> np.ndarray:
    t = z.flatten()
    return np.array([[1.0] if a >= 0 else [0.0] for a in t])


def res(x: int, y: int) -> int:
    return 1 if x * y >= 0 else 0


def rand() -> float:
    return random.random() * 2.0 - 1.0


def generate_numerical_data(
    train_size: int,
    test_size: int,
    norm: int = 0,
) -> list[tuple]:
    training_data = []

    def generate():
        x, y = rand(), rand()
        if norm == 1:
            val = abs(x) + abs(y)
            x, y = x / val, y / val
        if norm == 2:
            val = math.sqrt(x * x + y * y)
            x, y = x / val, y / val
        return x, y

    for _ in range(train_size):
        x, y = generate()
        data = np.array([np.array([x]), np.array([y])])
        training_data.append((data, res(x, y)))

    test_data = []
    for _ in range(test_size):
        x, y = generate()
        data = np.array([np.array([x]), np.array([y])])
        test_data.append((data, res(x, y)))

    return training_data, test_data


def run_with_numerical_data() -> None:
    norm = 1
    epochs = 20

    def create_network(
        func,
        func_der,
        train_data,
        test_data,
        batch_size,
        learn_rate,
    ):
        net = Network(
            [2, 4, 1],
            func,
            func_der,
            lambda a: 0 if abs(a) < abs(a - 1) else 1,
        )
        res = net.start(train_data, epochs, batch_size, learn_rate, test_data)
        return res

    training_data, test_data = generate_numerical_data(8000, 1000, norm)
    result = create_network(
        sigmoid, sigmoid_derivative, training_data, test_data, 4, 0.1
    )
    print(f"sigmoid {norm}: {result}")

    training_data, test_data = generate_numerical_data(8000, 1000, norm)
    result2 = create_network(ReLU, ReLU_derivative, training_data, test_data, 5, 0.1)
    print(f"ReLu {norm}: {result2}")


def run_with_image_data() -> None:
    epochs = 20

    import mnist_loader
    import numpy as np

    train_data, _, test_data = mnist_loader.load_data_wrapper()
    net = Network([28 * 28, 128, 10], sigmoid, sigmoid_derivative, np.argmax)
    res = net.start(train_data, epochs, 20, 1.0, test_data)
    print(f"MNIST DATA: {res}")


if __name__ == "__main__":
    run_with_numerical_data()
    run_with_image_data()
