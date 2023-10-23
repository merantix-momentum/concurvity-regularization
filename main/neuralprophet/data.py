from __future__ import annotations

import importlib
import logging

import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_dataset(dataset_cfg: DictConfig | dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load dataset and return a dataframe with the correct column names."""

    external_regressor_columns = dataset_cfg.get("external_regressor_columns", [])
    event_columns = dataset_cfg.get("event_columns", [])
    target_column = dataset_cfg["target_column"]
    date_column = dataset_cfg["date_column"]
    dataset_id_column = dataset_cfg.get("dataset_id_column", None)
    dataset = dataset_cfg["dataset"]
    normalize = dataset_cfg.get("normalize", False)

    if isinstance(dataset, str):
        # Convert to CSVDataset by loading import from str using importlib
        module, attr = dataset.rsplit(".", 1)
        dataset = getattr(importlib.import_module(module), attr)

    df = dataset.open()

    assert date_column in df.columns, "Dates not found in dataset."
    assert target_column in df.columns, "Target column not found in dataset."
    df = df.rename(columns={date_column: "ds", target_column: "y"})

    if dataset_id_column is not None:
        assert dataset_id_column in df.columns, "Dataset ID column not found in dataset."
        df = df.rename(columns={dataset_id_column: "dataset_id"})

    df.ds = pd.to_datetime(df.ds)

    df = df[["ds", "y"] + external_regressor_columns + event_columns + (["split"] if "split" in df.columns else [])]
    if dataset_cfg.create_split:
        df["split"] = (
            ["train"] * int(len(df) * 0.6)
            + ["val"] * int(len(df) * 0.2)
            + ["test"] * (int(len(df)) - int(len(df) * 0.6) - int(len(df) * 0.2))
        )
    if normalize:
        if "split" not in df.columns:
            scale_df = df
            logger.warning("Split column not found in dataset. Normalizing on entire dataset.")
        else:
            scale_df = df[df["split"] == "train"]

        df[["y"]] = StandardScaler(scale_df[["y"]]).transform(df[["y"]])

    return df, external_regressor_columns, event_columns


def construct_event_df(df: pd.DataFrame, event_columns: list[str]) -> pd.DataFrame | None:
    """Construct event dataframe from dataframe with event columns and reduce to daily frequency."""
    if len(event_columns) == 0:
        return None
    event_df = df[["ds"] + event_columns]
    event_df = event_df.groupby(event_df.ds.dt.date).first()
    event_df = pd.concat([pd.DataFrame({"event": col, "ds": event_df[event_df[col] == 1].ds}) for col in event_columns])
    return event_df


class StandardScaler:
    """Standardize data to have mean 0 and std 1."""

    def __init__(self, train_data: pd.DataFrame):
        self.mean = train_data.mean(0)
        self.std = train_data.std(0)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data to have mean 0 and std 1."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data to have original mean and std."""
        return data * self.std + self.mean
