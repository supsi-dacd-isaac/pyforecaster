from . import forecaster
from forecasting_models.holtwinters import HoltWinters, HoltWintersMulti

FORECASTER_MAP = {'Linear': forecaster.LinearForecaster,
                  'LGB': forecaster.LGBForecaster,
                  'HoltWinters': HoltWinters,
                  'HoltWintersMulti': HoltWintersMulti
                  }