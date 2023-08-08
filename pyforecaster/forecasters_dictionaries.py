from pyforecaster import forecaster
from pyforecaster.forecasting_models.holtwinters import HoltWinters, HoltWintersMulti
from pyforecaster.forecasting_models.gradientboosters import LGBMHybrid
from pyforecaster.forecasting_models.randomforests import QRF

FORECASTER_MAP = {'Linear': forecaster.LinearForecaster,
                  'LGB': forecaster.LGBForecaster,
                  'HoltWinters': HoltWinters,
                  'HoltWintersMulti': HoltWintersMulti,
                  'LGBHybrid': LGBMHybrid,
                  'QRF': QRF,
                  }