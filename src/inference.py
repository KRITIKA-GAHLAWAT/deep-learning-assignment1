"""Inference/evaluation entrypoint for Fashion-MNIST test set."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on Fashion-MNIST.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default="./models/model.npz")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"Implement inference pipeline with args: {args}")


if __name__ == "__main__":
    main()
