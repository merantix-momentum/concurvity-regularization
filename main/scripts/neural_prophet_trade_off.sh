#!/bin/bash

# Use log-uniform grid of 50 regularization strengths
REG_STRENGTHS=$(python -c "import numpy as np; print(','.join([str(x) for x in np.logspace(-6, 1, num=50, endpoint=True, base=10.0)]))")
SEEDS=$(python -c "print(','.join([str(x) for x in range(10)]))")
CF_NAME=train_synthetic_step_train
WANDB_JOB_TYPE=neural_prophet_synthetic_step_trade_off

# Generate runs for the tradeoff curve
python -m mxm_research_sandbox.concurvity.main.neural_prophet.train --config-name $CF_NAME \
  --multirun model_cfg.concurvity_reg_lambda=$REG_STRENGTHS seed=$SEEDS  ++wandb.job_type=$WANDB_JOB_TYPE hydra/launcher=joblib -p hydra.launcher|| exit
