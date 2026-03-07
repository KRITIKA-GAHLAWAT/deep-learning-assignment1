"""Data loading and preprocessing utilities for MNIST/Fashion-MNIST."""

from __future__ import annotations

from typing import Tuple

import numpy as np


Array = np.ndarray


def load_mnist(train: bool = True, data_dir: str = "./data") -> Tuple[Array, Array]:
    """Return MNIST images and labels."""
    raise NotImplementedError("Implement MNIST loading.")


def load_fashion_mnist(train: bool = False, data_dir: str = "./data") -> Tuple[Array, Array]:
    """Return Fashion-MNIST images and labels."""
    raise NotImplementedError("Implement Fashion-MNIST loading.")


def train_val_split(
    x: Array,
    y: Array,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Array, Array, Array, Array]:
    """Split arrays into train and validation sets."""
    raise NotImplementedError("Implement train/val split.")


def normalize_images(x: Array) -> Array:
    """Normalize pixel values to [0, 1]."""
    raise NotImplementedError("Implement image normalization.")


def flatten_images(x: Array) -> Array:
    """Flatten image tensors into 2D feature matrix."""
    raise NotImplementedError("Implement image flattening.")


def one_hot_encode(y: Array, num_classes: int = 10) -> Array:
    """Convert integer labels to one-hot vectors."""
    raise NotImplementedError("Implement one-hot encoding.")


def prepare_datasets(
    data_dir: str = "./data",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Prepare MNIST train/val and Fashion-MNIST test datasets."""
    raise NotImplementedError("Implement end-to-end dataset preparation.")
