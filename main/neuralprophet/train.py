from __future__ import annotations

import os
from tempfile import TemporaryDirectory

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning
import torchmetrics
import wandb
from omegaconf import DictConfig

from main.wandb_wrapper import init as wandb_init
from main.neuralprophet.data import construct_event_df, load_dataset
from main.neuralprophet.neuralprophet import ConcurvityNeuralProphet
from vendor.neuralprophet import NeuralProphet
from vendor.neuralprophet import utils as neuralprophet_utils


def construct_model(
    model_config: dict,
    event_columns: list[str],
    external_regressor_columns: list[str],
) -> NeuralProphet:
    model = ConcurvityNeuralProphet(
        **model_config,
        collect_metrics={
            "MSE": torchmetrics.MeanSquaredError(),
            "MAE": torchmetrics.MeanAbsoluteError(),
            "MAPE": torchmetrics.MeanAbsolutePercentageError(),
        },
    )
    if len(event_columns) > 0:
        model.add_events(event_columns)
    for col in external_regressor_columns:
        model.add_future_regressor(col)
    return model


def fit(
    model: NeuralProphet,
    df_train: pd.DataFrame,
    train_cfg: dict,
    early_stopping_target: str | None = None,
    run: wandb.wandb_sdk.wandb_run.Run | None = None,
    df_val: pd.DataFrame | None = None,
    val_plot_length: int | None = 84,
) -> tuple[NeuralProphet, pd.DataFrame]:
    """Train single prophet model and log validation prediction to wandb."""
    results = model.fit(
        df_train,
        validation_df=df_val,
        early_stopping_target=early_stopping_target,
        **train_cfg,
    )
    print(model.trainer.progress_bar_metrics)

    if df_val is not None and run:
        df_val_slice = df_val[:val_plot_length]
        prediction = model.predict(df_val_slice)
        model.plot(prediction)
        run.log({"Prediction (Validation)": wandb.Image(plt.gcf())})
        plt.close()

        dataset_ids = df_val_slice["ID"].unique().tolist() if "ID" in df_val_slice.columns else ["__df__"]
        for id in dataset_ids:
            model.plot_components(prediction, df_name=id)
            run.log({"Prophet components": wandb.Image(plt.gcf())})
            plt.close()

    return model, results


def save_model(
    model: NeuralProphet,
) -> None:
    """Save model as a W&B artifact"""
    with TemporaryDirectory() as dir:
        model_fp = os.path.join(dir, "model.np")
        neuralprophet_utils.save(model, model_fp)
        wandb.log_artifact(model_fp, name="model", type="model")


@hydra.main(config_path="../../configs/neuralprophet", config_name="train", version_base=None)
def main(cfg: DictConfig | dict):
    if seed := cfg.get("seed"):
        pytorch_lightning.seed_everything(seed)

    with wandb_init(config=cfg, **cfg.wandb) as run:
        dataset_cfg = cfg["dataset_cfg"]
        model_cfg = cfg["model_cfg"]
        train_cfg = cfg["train_cfg"]

        df, external_regressor_columns, event_columns = load_dataset(dataset_cfg)
        splits = df["split"]
        del df["split"]

        model = construct_model(model_cfg, event_columns, external_regressor_columns)
        event_df = construct_event_df(df, event_columns)
        if event_df is not None:
            df = model.create_df_with_events(df, events_df=event_df)

        df_train = df[splits == "train"]
        df_val = df[splits == "val"]

        model, results = fit(
            model,
            df_train=df_train,
            train_cfg=train_cfg,
            df_val=df_val,
            run=run,
            early_stopping_target="MAPE_val",
            val_plot_length=dataset_cfg.val_plot_length,
        )

        if run:
            save_model(model)


if __name__ == "__main__":
    main()
