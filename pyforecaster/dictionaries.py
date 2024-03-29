from . import copula
from . import tree_builders


def lgb_param_space(trial):
    param_space = {'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                   'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2)}
    return param_space


def linear_param_space(trial):
    param_space = {'kind': trial.suggest_categorical('kind', ['linear', 'ridge'])}
    return param_space

def picnn_param_space(trial):
    param_space = {'n_layers': trial.suggest_int('n_layers', 2, 5),
                   'n_hidden_x': trial.suggest_int('n_hidden_x', 50, 200),
                   'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1)
                   }

    return param_space

COPULA_MAP = {'HourlyGaussianCopula': copula.HourlyGaussianCopula,
              'ConditionalGaussianCopula': copula.ConditionalGaussianCopula}

TREE_MAP = {'DiffTree': tree_builders.DiffTree,
            'NeuralGas': tree_builders.NeuralGas,
            'ScenredTree': tree_builders.ScenredTree,
            'QuantileTree': tree_builders.QuantileTree}

HYPERPAR_MAP = {'LinearForecaster': linear_param_space,
                'LGBForecaster': lgb_param_space,
                'LGBMHybrid': lgb_param_space,
                "PICNN": picnn_param_space,
                "StructuredPICNN": picnn_param_space}

