from typing import List

from main.neuralprophet.time_net import ConcurvityTimeNet
from vendor.neuralprophet import log, NeuralProphet


class ConcurvityNeuralProphet(NeuralProphet):
    def __init__(self, *args, eta_min: float, concurvity_reg_lambda: float, concurvity_implementation: str, **kwargs):
        self.eta_min = eta_min
        self.concurvity_reg_lambda = concurvity_reg_lambda
        self.concurvity_implementation = concurvity_implementation
        super().__init__(*args, **kwargs)

    def _init_model(self):
        """Build Pytorch model with configured hyperparamters.

        Returns
        -------
            TimeNet model
        """
        self.model = ConcurvityTimeNet(
            config_train=self.config_train,
            config_trend=self.config_trend,
            config_ar=self.config_ar,
            config_seasonality=self.config_seasonality,
            config_lagged_regressors=self.config_lagged_regressors,
            config_regressors=self.config_regressors,
            config_events=self.config_events,
            config_holidays=self.config_country_holidays,
            config_normalization=self.config_normalization,
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            max_lags=self.max_lags,
            num_hidden_layers=self.config_model.num_hidden_layers,
            d_hidden=self.config_model.d_hidden,
            metrics=self.metrics,
            id_list=self.id_list,
            num_trends_modelled=self.num_trends_modelled,
            num_seasonalities_modelled=self.num_seasonalities_modelled,
            meta_used_in_model=self.meta_used_in_model,
            eta_min=self.eta_min,
            concurvity_reg_lambda=self.concurvity_reg_lambda,
            concurvity_implementation=self.concurvity_implementation,
        )
        log.debug(self.model)
        return self.model
