#!/bin/bash

# Use log-uniform grid of 50 regularization strengths
REG_STRENGTHS=$(python -c "import numpy as np; print(','.join([str(x) for x in np.logspace(-6, 1, num=50, endpoint=True, base=10.0)]))")
SEEDS=$(python -c "print(','.join([str(x) for x in range(10)]))")
CF_NAME=$1_train
WANDB_JOB_TYPE=$1_trade_off

# Generate runs for the tradeoff curve
python -m mxm_research_sandbox.concurvity.main.nam.train --config-name $CF_NAME \
  --multirun train_cfg.concurvity_reg_lambda_pairwise=$REG_STRENGTHS model_cfg.seed=$SEEDS  ++wandb.job_type=$WANDB_JOB_TYPE hydra/launcher=joblib -p hydra.launcher|| exit
