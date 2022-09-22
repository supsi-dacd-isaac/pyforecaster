from . import copula

COPULA_MAP = {'HourlyGaussianCopula': copula.HourlyGaussianCopula,
              'ConditionalGaussianCopula': copula.ConditionalGaussianCopula}
