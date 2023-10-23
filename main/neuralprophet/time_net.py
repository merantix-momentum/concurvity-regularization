import logging
import math

import torch

from main import concurvity
from vendor.neuralprophet.time_net import TimeNet

log = logging.getLogger("NP.concurvity_time_net")


class ConcurvityTimeNet(TimeNet):
    """Modification of neuralprophets time net with a concurvity based regularizer added."""

    def __init__(
        self,
        *args,
        eta_min: float = 1e-9,
        concurvity_reg_lambda: float,
        concurvity_implementation: str,
        use_delayed_concurvity: bool = False,
        **kwargs,
    ):
        """
        Args:
            eta_min: minimum learning rate
            concurvity_reg_lambda: lambda for the concurvity regularizer loss term
            concurvity_implementation:
            use_delayed_concurvity: whether to apply delayed regularization to concurvity as well.
        """
        super().__init__(*args, **kwargs)
        self.eta_min = eta_min

        # Regularization
        self.concurvity_reg_lambda = concurvity_reg_lambda
        self.concurvity_implementation = concurvity_implementation
        self.use_delayed_concurvity_reg = use_delayed_concurvity
        self.reg_enabled = self.reg_enabled or self.concurvity_reg_lambda

    def configure_optimizers(self):
        # Optimizer
        optimizer = self._optimizer(self.parameters(), lr=self.learning_rate, **self.config_train.optimizer_args)

        # Scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def loss_func(self, inputs, predicted, targets, concurvity_reg=None):
        loss = None
        # Compute loss. no reduction.
        loss = self.config_train.loss_func(predicted, targets)
        # Weigh newer samples more.
        loss = loss * self._get_time_based_sample_weight(t=inputs["time"])
        loss = loss.sum(dim=2).mean()
        # Regularize.
        if self.reg_enabled:
            steps_per_epoch = math.ceil(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
            progress_in_epoch = 1 - ((steps_per_epoch * (self.current_epoch + 1) - self.global_step) / steps_per_epoch)

            # Compute concurvity measure
            # Requires a re-computation of the components, but allows for much simpler code
            components = self.compute_components(inputs, meta=None)
            if self.config_trend.growth == "off":
                # If the trend is constant, the trend component shouldn't be considered for concurvity
                del components["trend"]

            components = torch.stack(list(components.values()))
            if self.concurvity_implementation == "pairwise":
                concurvity_reg = concurvity.pairwise(components, kind="corr")
            elif self.concurvity_implementation == "one_vs_rest":
                concurvity_reg = concurvity.one_vs_rest(components, kind="corr")
            else:
                raise NotImplementedError(
                    f"Unknown concurvity regularizer implementation {self.concurvity_implementation}"
                )

            loss, reg_loss = self._add_batch_regularizations(
                loss, self.current_epoch, progress_in_epoch, concurvity_reg
            )

            logging_key = "Concurvity"
            if self.trainer.validating:
                logging_key += "_val"
            elif self.trainer.testing:
                logging_key += "_test"
            self.log(logging_key, concurvity_reg.detach().cpu(), **self.log_args)

        else:
            reg_loss = torch.tensor(0.0, device=self.device)
        return loss, reg_loss

    def _add_batch_regularizations(self, loss, epoch, progress, concurvity_reg: torch.Tensor = None):
        """Add regularization terms to loss, if applicable

        Parameters
        ----------
            loss : torch.Tensor, scalar
                current batch loss
            epoch : int
                current epoch number
            progress : float
                progress within the epoch, between 0 and 1

        Returns
        -------
            loss, reg_loss
        """

        loss, reg_loss = super()._add_batch_regularizations(loss, epoch, progress)
        if self.concurvity_reg_lambda > 0:
            delay_weight = (
                self.config_train.get_reg_delay_weight(epoch, progress) if self.use_delayed_concurvity_reg else 1
            )

            loss += delay_weight * concurvity_reg * self.concurvity_reg_lambda
            reg_loss += delay_weight * concurvity_reg * self.concurvity_reg_lambda

        return loss, reg_loss
