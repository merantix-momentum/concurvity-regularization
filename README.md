# [Curve Your Enthusiasm: Concurvity Regularization in Differentiable Generalized Additive Models](https://arxiv.org/abs/2305.11475)
Accepted for [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71571)


## Abstract:
> Generalized Additive Models (GAMs) have recently experienced a resurgence in popularity, particularly in high-stakes domains such as healthcare. GAMs are favored due to their interpretability, which arises from expressing the target value as a sum of non-linear functions of the predictors. Despite the current enthusiasm for GAMs, their susceptibility to concurvity - i.e., (possibly non-linear) dependencies between the predictors - has hitherto been largely overlooked. Here, we demonstrate how concurvity can severly impair the interpretability of GAMs and propose a remedy: a conceptually simple, yet effective regularizer which penalizes pairwise correlations of the non-linearly transformed feature variables. This procedure is applicable to any gradient-based fitting of differentiable additive models, such as Neural Additive Models or NeuralProphet, and enhances interpretability by eliminating ambiguities due to self-canceling feature contributions. We validate the effectiveness of our regularizer in experiments on synthetic as well as real-world datasets for time-series and tabular data. Our experiments show that concurvity in GAMs can be reduced without significantly compromising prediction quality, improving interpretability and reducing variance in the feature importances.


## Authors:
[Julien Siems](https://scholar.google.de/citations?user=rKgTTh8AAAAJ&hl=de) *, [Konstantin Ditschuneit](https://ditschuneit.de/konstantin/) *, [Winfried Ripken](https://winfried-ripken.github.io/) *, [Alma Lindborg](https://scholar.google.com/citations?user=IBrCbDoAAAAJ&hl=sv) *, [Maximilian Schambach](https://maxschambach.github.io/), Johannes Otterbach, [Martin Genzel](https://martingenzel.com/)

* equal contribution

## BibTeX
```
@inproceedings{concurvity_reg,
 author = {Siems, Julien and Ditschuneit, Konstantin and Ripken, Winfried and Lindborg, Alma and Schambach, Maximilian and Otterbach, Johannes and Genzel, Martin},
 booktitle = {Advances in Neural Information Processing Systems},
 publisher = {Curran Associates, Inc.},
 title = {Increasing Confidence in Adversarial Robustness Evaluations},
 url = {https://neurips.cc/virtual/2023/poster/71571},
 volume = {37},
 year = {2023}
}
```

## Requirements
- Python >= 3.9
## Dependencies
```
$ pip install -r requirements.in
```
## Experiments
Note, that we use weights and biases throughout our experiments. To use wandb, please rename the file.env file to .env and replace the project and wandb API key.
If you would like to run the experiments without wandb, please set the environment variable `WANDB_MODE=dryrun`.

In the following, we provide the commands to reproduce the results shown in our paper.

### Toy Example 1
#### Figure 2 (a)
```
$ python -m main.nam.train —-config-name=toy_example_1_train
--multirun
dataset.sampling_correlation="0.0,1.0"
train_cfg.concurvity_reg_lambda_pairwise="0.0,1e-1"
model_cfg.seed="range(0, 40)"
wandb.group="toy_example_1_figure_2_a"
hydra/launcher=joblib
-p
hydra.launcher
```

#### Figure 2 (b)
```
$ ./main/scripts/nam_trade_off.sh toy_example_1
```

### Toy Example 2
#### Figure 3 (a)
```
$ python -m main.nam.train —-config-name=toy_example_2_train
--multirun
train_cfg.concurvity_reg_lambda_pairwise="0.0,1e-1"
model_cfg.seed="range(0, 40)"
wandb.group="toy_example_2_figure_3_a"
hydra/launcher=joblib
-p
hydra.launcher
```
#### Figure 3 (b)
```
$ ./main/scripts/nam_trade_off.sh toy_example_2
```

### Neural Prophet
#### Figure 4
```
$ ./main/scripts/neural_prophet_trade_off.sh
```

### Tabular Datasets
#### Figure 5
```
$ ./main/scripts/nam_trade_off.sh mimic2_train
```
```
$ ./main/scripts/nam_trade_off.sh mimic3_train
```
```
$ ./main/scripts/nam_trade_off.sh adult_train
```
```
$ ./main/scripts/nam_trade_off.sh california_housing_train
```
```
$ ./main/scripts/nam_trade_off.sh support2_train
```
```
$ ./main/scripts/nam_trade_off.sh boston_housing_train
```

#### Figure 6
```
$ python -m main.nam.train —-config-name=california_housing_train
--multirun
train_cfg.concurvity_reg_lambda_pairwise="0.0,0.1"
model_cfg.seed="range(0, 60)"
wandb.group="california_housing_figure_5_6"
hydra/launcher=joblib
-p
hydra.launcher
```

### pyGAM
Carries out HPO and evaluation of pyGAM
```
$ python main/traditional_gam.py —-config-name=california_housing_train
```