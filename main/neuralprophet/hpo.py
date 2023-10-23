from __future__ import annotations

import hydra
import optuna
from omegaconf import DictConfig

from main.wandb_wrapper import init as wandb_init
from main.neuralprophet.data import construct_event_df, load_dataset
from main.neuralprophet.train import construct_model, fit


def eval_neural_prophet(cfg: DictConfig, target: str):
    params = {}
    params.update(cfg)
    params["target"] = target
    params["num_hidden_neurons_regressor"] = cfg.model_cfg.hidden_sizes[0]
    params["num_hidden_layers_regressor"] = len(cfg.model_cfg.hidden_sizes)

    df, external_regressor_columns, event_columns = load_dataset(cfg.dataset_cfg)
    df_splits = {name: group for name, group in df.groupby("split")}
    df_train, df_val = df_splits["train"].drop(["split"], axis=1), df_splits["val"].drop(["split"], axis=1)
    with wandb_init(params, **cfg.wandb) as run:
        model = construct_model(cfg.model_cfg, event_columns, external_regressor_columns)
        event_df = construct_event_df(df, event_columns)
        if event_df is not None:
            df_train = model.create_df_with_events(df_train, events_df=event_df)
            df_val = model.create_df_with_events(df_val, events_df=event_df)

        model, results = fit(
            model,
            df_train=df_train,
            df_val=df_val,
            train_cfg=cfg.train_cfg,
            run=run,
            early_stopping_target=cfg.hpo_cfg.hpo_target,
        )
        run.finish()

    return model, results


def optuna_hpo_eval(trial: optuna.trial, target: str, cfg: DictConfig):
    cfg.model_cfg.learning_rate = float(trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True))
    cfg.train_cfg.epochs = int(trial.suggest_int("epochs", 90, 150))
    cfg.model_cfg.batch_size = int(trial.suggest_int("batch_size", 64, 1024, log=True))
    cfg.model_cfg.normalize = trial.suggest_categorical("normalize", ["soft1", "standardize"])
    cfg.model_cfg.optimizer = trial.suggest_categorical("optimizer", ["adamw", "rmsprop"])
    cfg.model_cfg.seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
    cfg.model_cfg.weekly_seasonality = int(trial.suggest_int("weekly_seasonality", 40, 80))
    cfg.model_cfg.yearly_seasonality = int(trial.suggest_int("yearly_seasonality", 20, 30))
    cfg.model_cfg.daily_seasonality = int(trial.suggest_int("daily_seasonality", 10, 40))
    num_hidden_layers_regressor = int(trial.suggest_int("num_hidden_layers_regressor", 1, 10))
    num_hidden_neurons_regressor = int(trial.suggest_int("num_hidden_neurons_regressor", 15, 40))
    cfg.model_cfg.hidden_sizes = [num_hidden_neurons_regressor] * num_hidden_layers_regressor
    cfg.model_cfg.seasonality_reg = float(trial.suggest_float("seasonality_reg", 1e-2, 1, log=True))
    cfg.model_cfg.trend_reg = float(trial.suggest_float("trend_reg", 1e-3, 1e-1, log=True))
    cfg.model_cfg.n_changepoints = int(trial.suggest_int("n_changepoints", 50, 90))
    cfg.model_cfg.batch_norm_first = trial.suggest_categorical("batch_norm_first", [True, False])
    cfg.model_cfg.weight_decay = float(trial.suggest_float("weight_decay", 5e-3, 5e-2, log=True))
    cfg.model_cfg.eta_min = float(trial.suggest_float("eta_min", 1e-9, 1e-5, log=True))
    cfg.model_cfg.dropout = float(trial.suggest_float("dropout", 1e-9, 1e-5, log=True))

    model, metrics = eval_neural_prophet(cfg, target=target)
    print("HPO OUTPUT:", metrics)
    return metrics[f"{cfg.hpo_cfg.hpo_target}(best)"]


@hydra.main(config_path="../../configs/neuralprophet", config_name="hpo", version_base=None)
def optuna_hpo(cfg: DictConfig | dict):
    def print_callback(study: optuna.study, trial: optuna.trial):
        print(f"Current value: {trial.value}, Current params: {trial.params}")
        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

    assert (
        cfg.wandb.group is not None
    ), "Please manually provide a wandb group name by passing `wandb.group=<GROUP NAME>`"

    sampler = optuna.integration.BoTorchSampler(
        n_startup_trials=cfg.hpo_cfg.n_startup_trials,
    )

    study = optuna.create_study(direction="minimize", sampler=sampler, load_if_exists=True)
    study.optimize(
        lambda trial: optuna_hpo_eval(trial, target="residual_load", cfg=cfg),
        timeout=360000,
        callbacks=[print_callback],
    )


if __name__ == "__main__":
    optuna_hpo()
