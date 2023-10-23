from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StepDataset:
    """Create synthetic df with step-wise periodicity in the features."""

    n_days: int
    """Number of days to generate."""

    def open(self) -> pd.DataFrame:
        """Create and return dataframe."""
        df = pd.DataFrame(columns=["ds", "y"])
        df["ds"] = pd.date_range(start="2020-01-01", periods=self.n_days, freq="H")
        df["y"] = 0

        # Add step-wise periodicity / 1 on weekends / 0.5 between 9am and 5pm
        df.loc[df["ds"].dt.dayofweek.isin([5, 6]), "y"] = 1
        df.loc[(df["ds"].dt.hour >= 9) & (df["ds"].dt.hour <= 17), "y"] += 0.5

        return df


step_5000 = StepDataset(n_days=5000)
step_1000 = StepDataset(n_days=1000)
