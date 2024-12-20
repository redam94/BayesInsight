from bayesinsight import BayesInsightModel

__all__ = ["compute_model_contributions"]


def compute_model_contributions(
    model: BayesInsightModel,
):
    """Compute the contributions of each variable to the model"""
    variable_details = model.variable_details
    variable_contributions = {}

    try:
        posterior = model.trace.posterior
    except KeyError:
        raise ValueError("Model needs to be fitted")

    for variable in variable_details:
        var_name = variable.var_name
        variable_contributions[variable.variable_name] = posterior[var_name].mean(
            dim=("chain", "draw")
        )

    return variable_contributions
