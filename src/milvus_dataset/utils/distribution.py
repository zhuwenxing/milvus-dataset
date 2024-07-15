import numpy as np
import pandas as pd


def generate_scalar_distribution(expression: str, hit_rate: float, size: int):
    if expression.startswith('normal'):
        mean, std = map(float, expression[7:-1].split(','))
        data = np.random.normal(mean, std, size)
    elif expression.startswith('uniform'):
        low, high = map(float, expression[8:-1].split(','))
        data = np.random.uniform(low, high, size)
    else:
        raise ValueError(f"Unsupported distribution: {expression}")

    mask = np.random.random(size) < hit_rate
    data[~mask] = np.nan

    return pd.Series(data)
