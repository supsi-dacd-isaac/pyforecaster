from . import copula
from . import tree_builders

COPULA_MAP = {'HourlyGaussianCopula': copula.HourlyGaussianCopula,
              'ConditionalGaussianCopula': copula.ConditionalGaussianCopula}

TREE_MAP = {'DiffTree': tree_builders.DiffTree,
            'NeuralGas': tree_builders.NeuralGas,
            'ScenredTree': tree_builders.ScenredTree,
            'QuantileTree': tree_builders.QuantileTree}
