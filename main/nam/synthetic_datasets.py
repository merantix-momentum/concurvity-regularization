from __future__ import annotations

import numpy as np
import pandas as pd


def linear_correlated(
        n_features: int,
        n_tasks: int,
        n_datapoints: int,
        seed: int | None = 42,
        sampling_correlation: float = 0.0,
        weights: np.array = None,
        *args,
        **kwargs,
) -> tuple[pd.DataFrame, np.array]:
    """Sample features with linear contribution that are correlated."""
    if weights is None:
        weights = np.array([np.linspace(-1, 1 * i, n_features) for i in range(1, n_tasks + 1)])
    else:
        weights = weights.reshape((n_tasks, n_features))

    if seed is not None:
        np.random.seed(seed)

    inputs_correlated = np.array([np.linspace(-2, 2, n_datapoints) for _ in range(n_features)]).T
    inputs_uniform = np.random.uniform(-2, 2, size=(n_datapoints, n_features))
    inputs = sampling_correlation * inputs_correlated + (1 - sampling_correlation) * inputs_uniform

    targets = weights.dot(inputs.T)

    df = pd.DataFrame()
    for i in range(n_features):
        df[f"input_{i}"] = inputs[:, i]

    for i in range(n_tasks):
        df[f"target_{i}"] = targets[i, :]

    return df, weights


def uncorrelated_but_dependent(
        n_datapoints: int, seed: int | None = 42, *args, **kwargs
) -> tuple[pd.DataFrame, np.array]:
    """Martins example."""
    if seed is not None:
        np.random.seed(seed)

    input_uniform = np.random.normal(0, 2, size=(n_datapoints))
    input_abs = np.abs(input_uniform)

    df = pd.DataFrame()
    df["input_0"] = input_uniform
    df["input_1"] = input_abs
    df["target_0"] = input_abs

    return df, None
