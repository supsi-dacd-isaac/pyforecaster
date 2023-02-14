from . import copula
from . import tree_builders


def lgb_param_space(trial):
    param_space = {'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                   'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.2)}
    return param_space


def linear_param_space(trial):
    param_space = {'kind': trial.suggest_categorical('kind', ['linear', 'ridge'])}
    return param_space


COPULA_MAP = {'HourlyGaussianCopula': copula.HourlyGaussianCopula,
              'ConditionalGaussianCopula': copula.ConditionalGaussianCopula}

TREE_MAP = {'DiffTree': tree_builders.DiffTree,
            'NeuralGas': tree_builders.NeuralGas,
            'ScenredTree': tree_builders.ScenredTree,
            'QuantileTree': tree_builders.QuantileTree}

HYPERPAR_MAP = {'LinearForecaster': linear_param_space,
                'LGBForecaster': lgb_param_space,
                'LGBHybrid': lgb_param_space}

