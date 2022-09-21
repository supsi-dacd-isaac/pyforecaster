from . import copula
from . import forecaster as f

COPULA_MAP = {'HourlyGaussianCopula': copula.HourlyGaussianCopula,
              'ConditionalGaussianCopula': copula.ConditionalGaussianCopula}

FORECASTER_MAP = {'linear': f.LinearForecaster,
                  'lgb': f.LGBForecaster}