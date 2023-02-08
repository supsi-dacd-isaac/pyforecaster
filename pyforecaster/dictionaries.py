from . import copula
from . import tree_builders
from . import forecaster
from forecasting_models.holtwinters import HoltWinters, HoltWintersMulti


COPULA_MAP = {'HourlyGaussianCopula': copula.HourlyGaussianCopula,
              'ConditionalGaussianCopula': copula.ConditionalGaussianCopula}

TREE_MAP = {'DiffTree': tree_builders.DiffTree,
            'NeuralGas': tree_builders.NeuralGas,
            'ScenredTree': tree_builders.ScenredTree,
            'QuantileTree': tree_builders.QuantileTree}

FORECASTER_MAP = {'Linear': forecaster.LinearForecaster,
                  'LGB': forecaster.LGBForecaster,
                  'HoltWinters': HoltWinters,
                  'HoltWintersMulti': HoltWintersMulti
                  }