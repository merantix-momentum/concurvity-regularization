from __future__ import annotations

import itertools
import os
import random
from collections import defaultdict
from typing import Any, Hashable

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from main.wandb_wrapper import init as wandb_init
from omegaconf import DictConfig
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC
from tqdm import tqdm

from main.nam import synthetic_datasets
from main import concurvity
from main import tabular as tabular_datasets

# this is required for joblib to work with wandb
os.environ["WANDB_START_METHOD"] = "thread"
sns.set_style("whitegrid")
font = {"family": "serif", "size": 9.5}
matplotlib.rc("font", **font)
plt.rcParams["figure.constrained_layout.use"] = True


class FeatureNN(torch.nn.Module):
    def __init__(self, hidden_sizes: list, activation_func: str):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.hidden_sizes = [1] + list(hidden_sizes) + [1]
        self.nn = torch.nn.ModuleList(
            torch.nn.Linear(before, after) for before, after in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])
        )
        self.activation_func = {
            "elu": torch.nn.functional.elu,
            "relu": torch.nn.functional.relu,
            "gelu": torch.nn.functional.gelu,
        }[activation_func]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.nn):
            x = layer(x)
            if layer_idx < len(self.nn) - 1:
                x = self.activation_func(x)
        return x


class NAM(torch.nn.Module):
    def __init__(self, hidden_sizes: list, features: list, activation_func="elu"):
        super().__init__()
        self.nns = torch.nn.ParameterDict(
            {f: FeatureNN(hidden_sizes=hidden_sizes, activation_func=activation_func) for f in features}
        )

        # Use the initialization of the linear layer, but move the layer to dict to ensure correct feature to
        # feature nn mapping.
        final_linear_layer = torch.nn.Linear(len(features), 1)
        self.final_linear_layer = torch.nn.ParameterDict(
            {f: final_linear_layer.weight.flatten()[f_idx] for f_idx, f in enumerate(features)}
        )
        self.bias = torch.nn.Parameter(final_linear_layer.bias)

    def forward(self, x: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # Accumulate NAM output
        feature_nn_outputs = {}
        for feature_name, nn in self.nns.items():
            feature_nn_outputs[feature_name] = (
                    nn(x[feature_name].reshape(-1, 1).to(torch.float32)).flatten() * self.final_linear_layer[
                feature_name]
            )
        return feature_nn_outputs, torch.sum(torch.stack(list(feature_nn_outputs.values())), axis=0) + self.bias


class PandasDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_names: list, target_names: list):
        self.feature_names: list = feature_names
        self.target_names: list = target_names
        self.df: pd.DataFrame = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        data = dict(self.df.iloc[idx, :]).items()
        return {k: v for k, v in data if k in self.feature_names}, {k: v for k, v in data if k in self.target_names}


def plot_shape_functions_toy_example(model: NAM, dataloader: DataLoader):
    inputs = defaultdict(list)
    contributions = defaultdict(list)
    targets = []
    with torch.no_grad():
        for features, target in tqdm(dataloader, desc="Evaluating the NAM for the dataset to draw shape functions."):
            contributions_per_batch, _ = model(features)
            for feature_name in features.keys():
                contributions[feature_name].append(contributions_per_batch[feature_name].detach())

            for feature_name, feature in features.items():
                inputs[feature_name].append(feature.detach())

            targets.extend(list(target.values()))

    contributions = {feature: torch.concat(contrib) for feature, contrib in contributions.items()}
    inputs = {feature: torch.concat(input_i) for feature, input_i in inputs.items()}

    # Pair Plot
    df = pd.DataFrame()
    df["target"] = targets[0].numpy()
    for feature_name, contrib in contributions.items():
        df[f"f_{feature_name[-1]}({feature_name})"] = contrib.numpy()

    sns.pairplot(df, plot_kws=dict(marker="+", linewidth=0.1))
    wandb.log({"Pair plot": wandb.Image(plt.gcf())})
    plt.close()

    # Compute Feature Importance.
    feature_importances = {}
    for feature, contrib in contributions.items():
        feature_importances[f"importance/{feature}"] = (contrib - contrib.mean()).abs().mean()
    wandb.log(feature_importances)

    # Draw shape functions
    fig, axs = plt.subplots(nrows=max(1, int(len(contributions) / 3)), ncols=min(len(contributions), 3), figsize=(9, 9))
    for ax, (feature, contrib) in zip(axs.flatten(), contributions.items()):
        ax.scatter(inputs[feature], contrib.flatten() - contrib.flatten().mean(), s=2, alpha=0.7)
        ax.set_title(feature)
    fig.suptitle(f"NAM Shape Functions | Beta: {float(model.bias.detach()):.3f}")
    wandb.log({"NAM Shape Functions (old)": wandb.Image(fig)})


def compute_loss_and_metrics(
        use_conc_reg: bool,
        cfg: DictConfig,
        targets: dict[str, torch.Tensor],
        target_names: list[str],
        feature_names: list[str],
        nam_output: torch.Tensor,
        feature_nn_outputs: dict[str, torch.Tensor],
        prefix: str,
        scalers: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metric_dict = {}
    if cfg.dataset.type == "regression":
        training_objective = torch.nn.functional.mse_loss(
            nam_output.reshape(-1, 1), targets[target_names[0]].reshape(-1, 1).to(torch.float32)
        )
        output_org_range = scalers[target_names[0]].inverse_transform(nam_output.reshape(-1, 1).detach().numpy())
        target_org_range = scalers[target_names[0]].inverse_transform(targets[target_names[0]].reshape(-1, 1).numpy())
        mse_org_range = metrics.mean_squared_error(target_org_range, output_org_range)
        rmse_org_range = np.sqrt(mse_org_range)

        metric_dict.update(
            {
                f"{prefix}/mse": training_objective,
                f"{prefix}/rmse": torch.sqrt(training_objective),
                f"{prefix}/rmse_org_range": rmse_org_range,
                f"{prefix}/mse_org_range": mse_org_range,
            }
        )
    else:
        training_objective = torch.nn.functional.binary_cross_entropy_with_logits(
            nam_output.reshape(-1, 1), targets[target_names[0]].reshape(-1, 1).to(torch.float32)
        )

        # Calculate AUC of precision - recall curve.
        pred = torch.sigmoid(nam_output.reshape(-1, 1)).detach().numpy()
        y = targets[target_names[0]].reshape(-1, 1).to(torch.float32)
        precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
        pr_auc = metrics.auc(recall, precision)

        metric_dict.update(
            {
                f"{prefix}/bce": training_objective,
                f"{prefix}/roc_auc": AUROC(task="binary")(
                    torch.sigmoid(nam_output.reshape(-1, 1)), targets[target_names[0]].reshape(-1, 1).to(torch.float32)
                ),
                f"{prefix}/pr_auc": pr_auc,
            }
        )
    stacked_feature_nn_out = torch.stack(list(feature_nn_outputs.values()))
    if use_conc_reg:
        pairwise_concurvity = concurvity.pairwise(
            stacked_feature_nn_out, kind=cfg.train_cfg.concurvity_reg_kind, eps=cfg.train_cfg.concurvity_reg_eps
        )
    else:
        pairwise_concurvity = 0
    loss = training_objective + cfg.train_cfg.concurvity_reg_lambda_pairwise * pairwise_concurvity

    metric_dict.update(
        {
            f"{prefix}/loss": loss,
            f"{prefix}/concurvity/pairwise_concurvity_cov": concurvity.pairwise(
                stacked_feature_nn_out, kind="cov", eps=cfg.train_cfg.concurvity_reg_eps
            ),
            f"{prefix}/concurvity/pairwise_concurvity_corr": concurvity.pairwise(
                stacked_feature_nn_out, kind="corr", eps=cfg.train_cfg.concurvity_reg_eps
            ),
        }
    )

    function_map = {
        "corr": lambda x: concurvity.correlation(
            torch.stack([x, targets[target_names[0]].reshape(-1, 1)]).reshape(2, -1)
        )[0, 1],
        "cov": lambda x: torch.cov(torch.stack([x, targets[target_names[0]].reshape(-1, 1)]).reshape(2, -1))[0, 1],
        "std": lambda x: x.std(),
    }
    for metric, feature_name in itertools.product(["cov", "corr", "std"], feature_names):
        metric_dict[f"{prefix}/{metric}/{feature_name}"] = function_map[metric](
            feature_nn_outputs[feature_name].reshape(-1, 1)
        )

    return loss, metric_dict


def fit_transform_df(
        df_train: pd.DataFrame, df_full: pd.DataFrame, cfg: DictConfig, skipcols: list = None
) -> tuple[DataFrame, dict[Hashable, OrdinalEncoder | StandardScaler | MinMaxScaler | RobustScaler]]:
    scalers = {}

    # Preprocess columns.
    for col_name, col in df_train.items():
        if skipcols is not None and col_name in skipcols:
            continue

        if is_string_dtype(col):
            scaler = OrdinalEncoder()
            # fit ordinal encoder to full dataset to include all categorical values.
            scaler.fit(df_full[col_name].values.reshape(-1, 1))
            df_train[col_name] = scaler.transform(col.values.reshape(-1, 1))
        elif is_numeric_dtype(col):
            if cfg.dataset.numeric_scaler == "standard":
                scaler = StandardScaler()
            elif cfg.dataset.numeric_scaler == "minmax":
                scaler = MinMaxScaler(feature_range=(-1, 1))
            elif cfg.dataset.numeric_scaler == "robust":
                scaler = RobustScaler()
            else:
                raise NotImplementedError(f"Unknown scaler {cfg.dataset.numeric_scaler}")

            df_train[col_name] = scaler.fit_transform(col.values.reshape(-1, 1))
        else:
            raise NotImplementedError(f"Unknown dtype error: {col_name.dtype}")

        scalers[col_name] = scaler

    return df_train, scalers


def transform_df(df: pd.DataFrame, scalers: dict, skipcols=None):
    # Preprocess columns.
    for col_name, col in df.items():
        if skipcols is not None and col_name in skipcols:
            continue

        df[col_name] = scalers[col_name].transform(col.values.reshape(-1, 1))

    return df


def retrieve_dataset(
        cfg: DictConfig,
) -> tuple[
    tuple[PandasDataset, PandasDataset, PandasDataset, PandasDataset],
    list[Any],
    Any,
    dict[Hashable, OrdinalEncoder | StandardScaler | MinMaxScaler | RobustScaler],
]:
    synthtic_data = False
    if hasattr(synthetic_datasets, cfg.dataset.name):
        dataset_fn = getattr(synthetic_datasets, cfg.dataset.name)
        df_orig, _ = dataset_fn(
            n_features=cfg.dataset.n_features,
            n_datapoints=cfg.dataset.n_datapoints,
            weights=np.array(list(cfg.dataset.weights)),
            n_tasks=1,
            sampling_correlation=cfg.dataset.sampling_correlation,
        )
        synthtic_data = True
    elif hasattr(tabular_datasets, cfg.dataset.name):
        df_orig = getattr(tabular_datasets, cfg.dataset.name).open()
    else:
        raise NotImplementedError(f"Unknown dataset {cfg.dataset_name}.")

    df = df_orig.copy(deep=True)

    # Drop rows with missing information.
    df = df.dropna()
    print(f"{len(df)} rows of {len(df_orig)} rows remaining after removing missing values.")

    skipcols = []
    if cfg.dataset.type == "classification":
        # Overwrite the scaling of the target column in case of a binary classification problem.
        skipcols.append(cfg.dataset.target_names[0])

    train_plus_val_ratio = cfg.dataset.train_ratio - cfg.dataset.val_ratio
    if synthtic_data:
        test_ratio = 1.0 - train_plus_val_ratio
        train_df, test_df = train_test_split(df, test_size=test_ratio)
    else:
        train_df, test_df = df[df["test"] == 0], df[df["test"] == 1]

    train_df, val_df = train_test_split(train_df, test_size=(cfg.dataset.val_ratio / train_plus_val_ratio))

    train_df, scalers = fit_transform_df(train_df, df, cfg, skipcols=skipcols)
    val_df = transform_df(val_df, scalers, skipcols=skipcols)
    test_df = transform_df(test_df, scalers, skipcols=skipcols)

    if synthtic_data:
        target_names = [x for x in df.columns if "target" in x]
        feature_names = list(df.columns.drop(*target_names))
    else:
        target_names = cfg.dataset.target_names
        feature_names = cfg.dataset.features

    dataset_full = PandasDataset(transform_df(df, scalers, skipcols=skipcols), feature_names, target_names)
    dataset_train = PandasDataset(train_df, feature_names, target_names)
    dataset_val = PandasDataset(val_df, feature_names, target_names)
    dataset_test = PandasDataset(test_df, feature_names, target_names)

    return (dataset_full, dataset_train, dataset_val, dataset_test), feature_names, target_names, scalers


@hydra.main(config_path="../../configs/nam")
def main(cfg: DictConfig | dict):
    cfg.model_cfg.seed = cfg.model_cfg.get("seed", random.randint(0, 1000))
    with wandb_init(config=cfg, **cfg.wandb) as run:
        # Seed, seed, seed
        seed = cfg.model_cfg.seed or random.randint(0, 1000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Init dataset and dataloaders
        (
            (full_dataset, train_dataset, val_dataset, test_dataset),
            feature_names,
            target_names,
            scalers,
        ) = retrieve_dataset(cfg)
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_cfg.batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

        # Record up to n steps per epoch
        total_log_steps = max(1, int(len(train_dataloader) / cfg.train_cfg.max_steps_to_track_per_epoch))

        # Init NAM
        hidden_sizes = cfg.model_cfg.get("hidden_sizes") or cfg.model_cfg.num_hidden * [cfg.model_cfg.hidden_dims]
        nam_model = NAM(hidden_sizes=hidden_sizes, features=feature_names, activation_func=cfg.model_cfg.activation)

        # Init optimizer and learning rate schedule
        optimizer = torch.optim.AdamW(
            nam_model.parameters(), lr=cfg.train_cfg.learning_rate, weight_decay=cfg.train_cfg.weight_decay
        )
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.train_cfg.num_epochs * len(train_dataloader)
        )

        val_training_objective = []
        global_step, total_num_steps = 0, len(train_dataloader) * cfg.train_cfg.num_epochs
        for epoch in range(cfg.train_cfg.num_epochs):
            # TRAINING
            for step_in_epoch, (features, targets) in enumerate(tqdm(train_dataloader, desc=f"Epoch: {epoch}")):
                optimizer.zero_grad()
                feature_nn_outputs, nam_output = nam_model(features)
                loss, metric_dict = compute_loss_and_metrics(
                    global_step > int(total_num_steps * cfg.train_cfg.concurvity_reg_start_step_ratio),
                    cfg,
                    targets,
                    target_names,
                    feature_names,
                    nam_output,
                    feature_nn_outputs,
                    prefix="train",
                    scalers=scalers,
                )
                # L1 Regularization on the final layer outputs.
                l1_reg = torch.sum(torch.stack([f_i.abs().mean() for f_i in feature_nn_outputs.values()]))
                # L1 regularization on final layer weights (feature selection)
                # l1_reg = torch.sum(
                #     torch.stack([v.abs() for v in nam_model.final_linear_layer.values()]))
                loss += cfg.train_cfg.l1_reg * l1_reg

                loss.backward()

                if step_in_epoch % total_log_steps == 0:
                    # Compute gradient magnitude.
                    grad_magnitude = 0
                    for param_name, param in nam_model.named_parameters():
                        grad_magnitude += (param.grad ** 2).sum()
                    metric_dict.update({"train/grad_magnitude": torch.sqrt(grad_magnitude)})
                    metric_dict.update({"train/l1_reg": l1_reg})
                    wandb.log(metric_dict)

                optimizer.step()
                lr_schedule.step()
                global_step += 1

                # Stop and return a high loss if the loss is NaN
                if torch.any(torch.isnan(loss)):
                    print("NAN ENCOUNTERED")
                    return 1e2

            # VALIDATION
            for features, targets in val_dataloader:
                feature_nn_outputs, nam_output = nam_model(features)
                _, metric_dict = compute_loss_and_metrics(
                    global_step > int(total_num_steps * cfg.train_cfg.concurvity_reg_start_step_ratio),
                    cfg,
                    targets,
                    target_names,
                    feature_names,
                    nam_output,
                    feature_nn_outputs,
                    prefix="val",
                    scalers=scalers,
                )
                wandb.log(metric_dict)
                val_training_objective.append(
                    metric_dict["val/mse" if cfg.dataset.type == "regression" else "val/bce"].item()
                )

        # TEST
        for features, targets in test_dataloader:
            feature_nn_outputs, nam_output = nam_model(features)
            _, metric_dict = compute_loss_and_metrics(
                global_step > int(total_num_steps * cfg.train_cfg.concurvity_reg_start_step_ratio),
                cfg,
                targets,
                target_names,
                feature_names,
                nam_output,
                feature_nn_outputs,
                prefix="test",
                scalers=scalers,
            )
            wandb.log(metric_dict)

        plot_shape_functions_toy_example(nam_model, DataLoader(full_dataset, batch_size=len(full_dataset)))
        plt.close()

        return (
            val_training_objective[-1] if ~np.isnan(val_training_objective[-1]) else 1e2
        )  # Return last validation training objective for e.g. HPO


if __name__ == "__main__":
    main()
