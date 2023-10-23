from __future__ import annotations

import contextlib
import os

import torch
import wandb
from dotenv import load_dotenv
from flatten_dict import flatten_dict
from omegaconf import DictConfig


# Not the nicest way, but should be fail-safe unless this file is moved
_SRC_DIR = os.path.normpath(os.path.dirname(__file__))
WANDB_DIR = os.path.join(_SRC_DIR, "wandb")

# File extensions to include in the code log
INCLUDE_EXTENSIONS = {".py", ".yaml", ".yml", ".txt", ".md", ".sh", ".ipynb", ".html"}


def _configure_settings(**kwargs) -> None:

    # Load environment variables from the file `/src/.env`.
    load_dotenv(dotenv_path=os.path.join(_SRC_DIR, ".env"))

    assert "WANDB_API_KEY" in os.environ, (
        "Please explicitly set the env variable `WANDB_API_KEY` or add it to the file `/src/.env`. "
        "This will enable W&B to work out of the box with kubernetes pods and flyte. "
        "You can find your API key on https://wandb.merantix-momentum.cloud/settings."
    )


def init(
    config: dict | DictConfig, enabled: bool = True, **kwargs
) -> wandb.wandb_sdk.wandb_run.Run | contextlib.nullcontext:
    """Wrapper around `wandb.init` that sets defaults, ensures correct usage of our wandb server and logs the complete
    config and code.

    The function uses `python-dotenv` to load environment variables out of the file `/src/.env`. The file is ignored by
    git, but synced to debug pods and flyte. You can seemlessly log into W&B on all machines by adding your W&B API key:

      WANDB_API_KEY=...

    Args:
        config: dictionary containing your run config.
        enabled: Boolean flag that can be used to disable wandb completely. Helpful for debugging purposes.
        kwargs: Arguments passed to wandb.init()

    Returns:
        Either a wandb run or a null context if enabled=False. In this case, run=None if the result is used as a context
        manager.
    """
    if not enabled:
        return contextlib.nullcontext()

    _configure_settings(**kwargs)

    assert "project" in kwargs, "Please set a project name."

    config = flatten_dict.flatten(config, reducer="path")

    # Check for and log available torch backends
    config.update(
        {
            f"torch.backends.{name}": backend.is_available()
            for name, backend in torch.backends.__dict__.items()
            if hasattr(backend, "is_available")
        }
    )

    # Make sure that the wandb dir is created
    os.makedirs(WANDB_DIR, exist_ok=True)

    run = wandb.init(config=config, dir=WANDB_DIR, **kwargs)

    # Log all code in /src
    run.log_code(".", include_fn=lambda x: any(x.endswith(ext) for ext in INCLUDE_EXTENSIONS))

    return run
