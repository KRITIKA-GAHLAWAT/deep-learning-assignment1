"""Simple dense layer utility for NumPy-based ANN code."""

from __future__ import annotations

import numpy as np


class NeuralLayer:
    """Dense layer container with forward pass helper."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weights = np.zeros((self.input_dim, self.output_dim), dtype=np.float64)
        self.bias = np.zeros((1, self.output_dim), dtype=np.float64)

    def initialize(self, method: str = "xavier", random_seed: int | None = None) -> None:
        rng = np.random.default_rng(random_seed)
        m = method.lower()

        if m == "zeros":
            self.weights.fill(0.0)
            self.bias.fill(0.0)
            return
        if m == "normal":
            self.weights = rng.standard_normal((self.input_dim, self.output_dim)) * 0.01
            self.bias.fill(0.0)
            return
        if m == "he":
            scale = np.sqrt(2.0 / self.input_dim)
            self.weights = rng.standard_normal((self.input_dim, self.output_dim)) * scale
            self.bias.fill(0.0)
            return
        if m == "xavier":
            scale = np.sqrt(1.0 / self.input_dim)
            self.weights = rng.standard_normal((self.input_dim, self.output_dim)) * scale
            self.bias.fill(0.0)
            return

        raise ValueError(f"Unsupported initialization method: {method}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

