from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig

from main.neuralprophet.data import construct_event_df, load_dataset
from main.neuralprophet.train import construct_model, fit
from main.wandb_wrapper import init as wandb_init


@hydra.main(config_path="../../configs/neuralprophet", config_name="train", version_base=None)
def main(cfg: DictConfig | dict):
    lambdas = np.linspace(-7, 2, 30)
    np.random.shuffle(lambdas)
    for concurvity_reg_lambda in lambdas:
        with wandb_init(config=cfg, group="concurvity_experiment_ercot_sweep_3", **cfg.wandb) as run:
            dataset_cfg = cfg["dataset_cfg"]
            model_cfg = cfg["model_cfg"]
            train_cfg = cfg["train_cfg"]

            model_cfg["concurvity_reg_lambda"] = float(10 ** concurvity_reg_lambda)

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
                model, df_train=df_train, train_cfg=train_cfg, df_val=df_val, run=run, early_stopping_target="MAPE_val"
            )
            run.finish()


if __name__ == "__main__":
    main()
