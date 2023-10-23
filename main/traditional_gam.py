from __future__ import annotations

import json
from collections import defaultdict
from datetime import date

import hydra
import numpy as np
import pandas as pd
import pygam
import torch
from numpyencoder import NumpyEncoder
from omegaconf import DictConfig
from pygam import LinearGAM, LogisticGAM, s
from sklearn import metrics
from torchmetrics import AUROC

from main import concurvity
from main.nam.train import retrieve_dataset


def compute_concurvity(gam: pygam.GAM, X: np.ndarray) -> float:
    feature_outputs = {}
    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue

        pdep, confi = gam.partial_dependence(term=i, X=X, width=0.95)
        feature_outputs[i] = pdep

    return float(
        concurvity.pairwise(torch.from_numpy(np.stack([v for v in feature_outputs.values()])), kind="corr", eps=1e-12)
    )


@hydra.main(config_path="../configs/nam", config_name="mimic2_train")
def main(cfg: DictConfig | dict):
    # Init dataset and dataloaders
    (
        (full_dataset, train_dataset, val_dataset, test_dataset),
        feature_names,
        target_names,
        scalers,
    ) = retrieve_dataset(cfg)

    train_val_df = pd.concat([train_dataset.df, val_dataset.df], axis=0)

    # Train - Val dataset
    X = train_val_df[full_dataset.feature_names].to_numpy()
    y = train_val_df[full_dataset.target_names].to_numpy()

    # Estimate absolute mean correlation.
    X_corr = np.corrcoef(X.T)
    X_corr = X_corr[np.triu_indices_from(X_corr, k=1)]
    X_corr[np.isnan(X_corr)] = 0.0
    abs_mean_corr = np.abs(X_corr).mean()
    print(f"{cfg.dataset.name}: Absolute Mean Correlation: {abs_mean_corr}")

    # Test dataset
    X_test = test_dataset.df[full_dataset.feature_names].to_numpy()
    y_test = test_dataset.df[full_dataset.target_names].to_numpy()

    if cfg.dataset.type == "classification":
        gam = LogisticGAM()
    else:
        gam = LinearGAM()
    lams = np.random.rand(10, X.shape[1])  # random points on [0, 1], with shape (100, 3)
    lams = lams * 8 - 4  # shift values to -3, 3
    lams = 10 ** lams  # transforms values to 1e-3, 1e3

    gam_hpo = gam.gridsearch(X, y, lam=lams)
    gam_hpo.summary()

    data = defaultdict(dict)
    data["coeff"] = gam_hpo.coef_
    for i, term in enumerate(gam_hpo.terms):
        if term.isintercept:
            continue

        XX = gam_hpo.generate_X_grid(term=i)
        pdep, confi = gam_hpo.partial_dependence(term=i, X=XX, width=0.95)
        xs = scalers[full_dataset.feature_names[i]].inverse_transform(XX[:, term.feature].reshape(-1, 1))

        data[full_dataset.feature_names[i]]["xs"] = xs.tolist()
        data[full_dataset.feature_names[i]]["pdep"] = pdep.tolist()
        data[full_dataset.feature_names[i]]["confi"] = confi.tolist()

    pred = gam_hpo.predict_mu(X_test)
    if cfg.dataset.type == "classification":
        test_roc_auc = float(AUROC(task="binary")(torch.from_numpy(pred), torch.from_numpy(y_test)))
        data["test_roc_auc"] = test_roc_auc
        test_bce = float(
            torch.nn.BCELoss()(torch.from_numpy(pred).flatten().float(), torch.from_numpy(y_test).flatten().float())
        )
        data["test_bce"] = test_bce
        print("HPO ROC AUC", test_roc_auc)
        print("HPO BCE", test_bce)
    else:
        pred_org_range = scalers[cfg.dataset.target_names[0]].inverse_transform(pred.reshape(-1, 1))
        target_org_range = scalers[cfg.dataset.target_names[0]].inverse_transform(y_test.reshape(-1, 1))
        test_rmse = np.sqrt(metrics.mean_squared_error(target_org_range, pred_org_range))
        data["test_rmse"] = float(test_rmse)
        print("HPO RMSE", test_rmse)

    data["concurvity"] = float(compute_concurvity(gam, X))
    print(f"Concurvity (Corr) Pairwise: {data['concurvity']}")
    json.dump(
        data, open(f"pygam_{cfg.dataset.name}.json", "w"), cls=NumpyEncoder
    )


if __name__ == "__main__":
    main()
