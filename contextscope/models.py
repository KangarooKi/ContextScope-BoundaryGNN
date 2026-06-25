from __future__ import annotations

import math


class LogisticRegressionGD:
    def __init__(
        self,
        epochs: int = 180,
        lr: float = 0.08,
        l2: float = 1e-4,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, x: list[list[float]], y: list[int], train_idx: list[int]) -> None:
        if not x:
            return
        width = len(x[0])
        self.weights = [0.0] * width
        pos_rate = (sum(y[idx] for idx in train_idx) + 0.5) / (len(train_idx) + 1.0)
        self.bias = logit(pos_rate)
        for _ in range(self.epochs):
            grad_w = [0.0] * width
            grad_b = 0.0
            for idx in train_idx:
                pred = sigmoid(dot(self.weights, x[idx]) + self.bias)
                error = pred - y[idx]
                row = x[idx]
                for col, value in enumerate(row):
                    grad_w[col] += error * value
                grad_b += error
            denom = max(1.0, float(len(train_idx)))
            for col in range(width):
                grad = grad_w[col] / denom + self.l2 * self.weights[col]
                self.weights[col] -= self.lr * grad
            self.bias -= self.lr * grad_b / denom

    def predict_proba(self, x: list[list[float]]) -> list[float]:
        return [sigmoid(dot(self.weights, row) + self.bias) for row in x]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(probability: float) -> float:
    clipped = min(max(probability, 1e-5), 1.0 - 1e-5)
    return math.log(clipped / (1.0 - clipped))
