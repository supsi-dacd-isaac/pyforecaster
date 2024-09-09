import numpy as np
import pandas as pd
from pyforecaster.forecaster import ScenarioGenerator
from statsmodels.tsa.statespace import exponential_smoothing
from tqdm import tqdm
from pyforecaster.forecasting_models.holtwinters import hankel

def get_forecasts(model, y_te, x_te, n_step_ahead=24):
  y_hat = np.zeros((len(y_te), n_step_ahead))
  for i in tqdm(range(len(y_te))):
    y_hat[i,:] = model.forecast(n_step_ahead)
    exog = x_te.iloc[[i]].values if x_te is not None else None
    try:
        model = model.append(y_te.iloc[[i]], exog=exog) # this method UPDATES the model with the last observation
    except:
        print('Error in updating the model')
        model = model.apply(y_te.iloc[[i]], exog=exog)  # this method UPDATES the model with the last observation
  return pd.DataFrame(y_hat, index=y_te.index)


class StatsModelsWrapper(ScenarioGenerator):
    def __init__(self, model_class, model_pars, target_name, q_vect, nodes_at_step=None, val_ratio=0.8, n_sa=1,
                 **scengen_kwgs):
        self.model_class = model_class
        self.model_pars = model_pars
        self.model = None
        self.target_name = target_name
        self.n_sa = n_sa
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, **scengen_kwgs)

    def fit(self, x_pd:pd.DataFrame, y:pd.DataFrame=None):
        y_present = x_pd[self.target_name].values

        # exclude the last n_sa, we need them to create the target
        preds = self.predict(x_pd)[:-self.n_sa]

        # hankelize the target
        hw_target = hankel(y_present[1:], self.n_sa)
        resid = hw_target - preds
        self.err_distr = np.quantile(resid, self.q_vect, axis=0).T


    def predict(self, x_pd:pd.DataFrame, **kwargs):
        pass

    def predict_quantiles(self, x:pd.DataFrame, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        for h in np.unique(x.index.hour):
            preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        return preds


class ExponentialSmoothing(StatsModelsWrapper):
    def __init__(self, target_name, q_vect, nodes_at_step=None, val_ratio=0.8, n_sa=1, trend=None, seasonal=None, **scengen_kwgs):
        model_class = exponential_smoothing.ExponentialSmoothing
        model_pars = {'trend': trend, 'seasonal': seasonal}
        super().__init__(model_class, model_pars, target_name, q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, n_sa=n_sa, **scengen_kwgs)

    def fit(self, x_pd:pd.DataFrame, y:pd.DataFrame=None):
        x_pd = x_pd.asfreq(pd.infer_freq(x_pd.index))
        y_pd = x_pd[[self.target_name]]
        self.model = self.model_class(**self.model_pars, endog=y_pd).fit()
        super().fit(x_pd, y_pd)
        return self

    def predict(self, x_pd:pd.DataFrame, **kwargs):
        x_pd = x_pd.asfreq(pd.infer_freq(x_pd.index))
        y = x_pd[self.target_name]
        y_hat = get_forecasts(self.model, y, n_step_ahead=self.n_sa, x_te=None)
        return y_hat

