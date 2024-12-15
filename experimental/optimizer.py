from bayesinsight import BayesInsightModel
from bayesinsight.models.variablemodels import VariableDetails

from typing import Tuple, Dict, List, Union, Iterable

import pandas as pd
import xarray as xr
import numpy as np
import pymc as pm


def objective(mu: xr.DataArray, dim: Tuple[str], model: BayesInsightModel) -> float:
    """Objective function for foottraffic model"""
    original_model_mu = model.trace.posterior.mu.sum(dim=dim).mean().values
    new_model_mu = mu.sum(dim=dim).mean().values
    return (-new_model_mu + original_model_mu) / original_model_mu


def get_scaling_factor(
    original_spend: float, new_spend: float, original_cpm: float, new_cpm: float
) -> float:
    """Scale for impressions"""
    original_imps = original_spend / original_cpm * 1000
    new_imps = new_spend / new_cpm * 1000
    return new_imps / original_imps


def weights_to_scaling_factors(
    x: List[float],
    new_budgets: Dict[str, float],
    new_cpms: Dict[str, float],
    original_budget: Dict[str, float],
    original_cpm: Dict[str, float],
) -> Dict[str, float]:
    """Scale impressions during the start and end period by
    the fraction more or less impressions the new budget would imply based on
    the old budget and cpms.
    This allows the solver to look through percentage increase/decrease bounds instead of actuall $ amounts.
    """
    scaling_factor = {
        key: get_scaling_factor(
            original_budget[key],
            new_budgets[key] * (1 + x[i]),
            original_cpm[key],
            new_cpms[key],
        )
        - 1
        for i, key in enumerate(original_budget.keys())
    }

    return scaling_factor


def make_input_mask(
    start_period: Union[str, pd.Timestamp],
    end_period: Union[str, pd.Timestamp],
    model_dates: Iterable[pd.Timestamp],
) -> xr.DataArray:
    """Treat periods outside the masked period as if they didn't change"""

    if isinstance(start_period, str):
        start_period = pd.to_datetime(start_period)

    if isinstance(end_period, str):
        end_period = pd.to_datetime(end_period)

    input_mask = xr.DataArray(
        np.ones_like(model_dates, dtype=float),
        dims=("Period",),
        coords={"Period": model_dates},
    ).where(lambda x: (x.Period > start_period) & (x.Period < end_period), 0)

    return input_mask


def compute_contributions(
    x: List[float],
    new_budget: Dict[str, float],
    new_cpm: Dict[str, float],
    start_period: str,
    end_period: str,
    model: BayesInsightModel,
    var_map: Dict[str, float],
    vars: List[VariableDetails],
) -> xr.Dataset:
    """Compute contributions from solution weights"""

    updated_weights = weights_to_scaling_factors(x, new_budget, new_cpm)
    input_mask = make_input_mask(start_period, end_period, model.trace.posterior.Period)
    pymc_model = model.build()

    with pymc_model:
        pm.set_data(
            {
                media_var.variable_name: (
                    (updated_weights[var_map[media_var.variable_name]] * input_mask + 1)
                    * model.trace.constant_data[media_var.variable_name]
                ).transpose(*model.trace.constant_data[media_var.variable_name].dims)
                for i, media_var in enumerate(vars)
            }
        )
        det_ = pm.compute_deterministics(model.trace.posterior)

    return det_


def non_heirarcical_bayesinsight_loss_function(
    x,
    new_budget,
    new_cpm,
    input_mask,
    var_map,
    model,
    vars: List[VariableDetails],
    external_variables=None,
    chains_to_use=slice(0, 1),
    draws_to_use=slice(0, 1000, 5),
):
    updated_weights = weights_to_scaling_factors(x, new_budget, new_cpm)
    pymc_model = model.build()
    with pymc_model:
        pm.set_data(
            {
                media_var.variable_name: (
                    (updated_weights[var_map[media_var.variable_name]] * input_mask + 1)
                    * model.trace.constant_data[media_var.variable_name]
                ).transpose(*model.trace.constant_data[media_var.variable_name].dims)
                for i, media_var in enumerate(vars)
            }
        )
        if external_variables is not None:
            pass
        det_ = pm.compute_deterministics(
            model.trace.posterior.sel(chain=chains_to_use, draw=draws_to_use),
            var_names=["mu"],
            progressbar=False,
        )
    return objective(det_.mu)


def eq_constraint(x: list, starting_budget: dict[str, float]):
    starting_budget_ = sum(starting_budget.values())
    updated_budget = sum(
        (1 + x[i]) * starting_budget[media_var]
        for i, media_var in enumerate(starting_budget.keys())
    )

    return (starting_budget_ - updated_budget) / starting_budget_
