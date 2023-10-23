from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pandas as pd

LOCAL_PREFIX: Path = Path(__file__).parent / "datasets"

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CSVDataset:
    """Basic dataset class."""

    subdir: str | Path
    name: str | Path

    def open(self) -> pd.DataFrame:
        """Open a timeseries dataset.

        Download the dataset if it does not exist locally.
        """

        local_path = LOCAL_PREFIX / self.subdir / f"{self.name}.csv"
        df = pd.read_csv(local_path)
        return df


SUBDIR = "tabular"

boston_housing = CSVDataset(SUBDIR, "boston_housing")
"""Boston housing dataset."""
california_housing = CSVDataset(SUBDIR, "california_housing")
"""California housing dataset."""
adult = CSVDataset(SUBDIR, "adult")
"""UCI adult dataset."""
mimic2 = CSVDataset(SUBDIR, "mimic2")
"""MIMIC3 dataset."""
mimic3 = CSVDataset(SUBDIR, "mimic3")
"""MIMIC3 dataset."""
support2 = CSVDataset(SUBDIR, "support2")
"""support2 dataset."""
