# MOST OF THIS CODE IS TAKEN FROM https://github.com/tiskw/random-fourier-features/blob/main/rfflearn/cpu/rfflearn_cpu_regression.py
# Author: Tetsuya Ishikawa

import functools
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
from pyforecaster.forecaster import ScenarioGenerator
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

class BrutalRegressor:
    def __init__(self, n_iter=10):
        self.best_pars = []
        self.n_iter = n_iter
        self.target_cols = None
    def fit_feature(self, x, y, n_bins=10):

        cuts = np.quantile(x, np.linspace(0, 1, n_bins+1))
        x_binned = np.digitize(x, cuts)-1
        y_means = np.array([y[x_binned == i].mean() for i in range(n_bins+1)])
        score = np.mean((y - y_means[x_binned, :])**2)**0.5
        pars = {'cuts': cuts, 'y_means': y_means, 'score': score}

        return pars

    def one_level_fit(self, x, y):
        with ProcessPoolExecutor(4) as executor:
            pars = list(executor.map(partial(self.fit_feature, y=y), [x[:, i] for i in range(x.shape[1])]))
        # find best feature
        best_feature = np.argmin([s['score'] for s in pars])
        best_pars = pars[best_feature]
        return best_feature, best_pars
    def fit(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.values
        self.target_cols = y.columns
        err = y.copy()
        for i in tqdm(range(self.n_iter)):
            best_feature, best_pars = self.one_level_fit(x, err)
            y_hat = best_pars['y_means'][np.digitize(x[:, best_feature], best_pars['cuts'])-1]
            err = err - y_hat
            print(np.abs(err).mean().mean())
            self.best_pars.append(best_pars)
        return self

    def predict(self, x):

        y_hat = np.zeros((x.shape[0], self.best_pars[0]['y_means'].shape[1]))
        for i in range(self.n_iter):
            y_hat += self.best_pars[i]['y_means'][np.digitize(x.iloc[:, i], self.best_pars[i]['cuts'])-1]
        return pd.DataFrame(y_hat, index = x.index, columns=self.target_cols)



def seed(seed):
    """
    Fix random seed used in this script.

    Args:
        seed (int): Random seed.
    """
    # Now it is enough to fix the random seed of Numpy.
    np.random.seed(seed)


def get_rff_matrix(dim_in, dim_out, std):
    """
    Generates random matrix of random Fourier features.

    Args:
        dim_in  (int)  : Input dimension of the random matrix.
        dim_out (int)  : Output dimension of the random matrix.
        std     (float): Standard deviation of the random matrix.

    Returns:
        (np.ndarray): Random matrix with shape (dim_out, dim_in).
    """
    return std * np.random.randn(dim_in, dim_out)


def get_orf_matrix(dim_in, dim_out, std):
    """
    Generates random matrix of orthogonal random features.

    Args:
        dim_in  (int)  : Input dimension of the random matrix.
        dim_out (int)  : Output dimension of the random matrix.
        std     (float): Standard deviation of the random matrix.

    Returns:
        (np.ndarray): Random matrix with shape (dim_out, dim_in).
    """
    # Initialize matrix W.
    W = None

    for _ in range(dim_out // dim_in + 1):
        s = scipy.stats.chi.rvs(df = dim_in, size = (dim_in, ))
        Q = np.linalg.qr(np.random.randn(dim_in, dim_in))[0]
        V = std * np.dot(np.diag(s), Q)
        W = V if W is None else np.concatenate([W, V], axis = 1)

    # Trim unnecessary part.
    return W[:dim_in, :dim_out]


def get_matrix_generator(rand_type, std, dim_kernel):
    """
    This function returns a function which generate RFF/ORF matrix.
    The usage of the returned value of this function are:
        f(dim_input:int) -> np.array with shape (dim_input, dim_kernel)
    """
    if   rand_type == "rff": return functools.partial(get_rff_matrix, std = std, dim_out = dim_kernel)
    elif rand_type == "orf": return functools.partial(get_orf_matrix, std = std, dim_out = dim_kernel)
    else                   : raise RuntimeError("matrix_generator: 'rand_type' must be 'rff', 'orf', or 'qrf'.")


class Base:
    """
    Base class of the following RFF/ORF related classes.
    """
    def __init__(self, rand_type, dim_kernel, std_kernel, W, b):
        """
        Constractor of the Base class.
        Create random matrix generator and random matrix instance.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            W          (np.ndarray): Random matrix for the input `X`. If None then generated automatically.
            b          (np.ndarray): Random bias for the input `X`. If None then generated automatically.

        Notes:
            If `W` is None then the appropriate matrix will be set just before the training.
        """
        self.dim = dim_kernel
        self.s_k = std_kernel
        self.mat = get_matrix_generator(rand_type, std_kernel, dim_kernel)
        self.W   = W
        self.b   = b

    def conv(self, x, index=None):
        """
        Applies random matrix to the given input vectors `X` and create feature vectors.

        Args:
            x     (pd.DataFrame): Input matrix with shape (n_samples, n_features).
            index (int)       : Index of the random matrix. This value should be specified only
                                when multiple random matrices are used.

        Notes:
            This function can manipulate multiple random matrix. If argument 'index' is given,
            then use self.W[index] as a random matrix, otherwise use self.W itself.
            Also, computation of `ts` is equivarent with ts = X @ W, however, for reducing memory
            consumption, split X to smaller matrices and concatenate after multiplication wit W.
        """
        W = self.W if index is None else self.W[index]
        b = self.b if index is None else self.b[index]
        return pd.DataFrame(np.cos(x.values @ W + b), index=x.index)

    def set_weight(self, dim_in):
        """
        Set the appropriate random matrix to 'self.W' if 'self.W' is None (i.e. empty).

        Args:
            dim_in (int): Input dimension of the random matrix.

        Notes:
            This function can manipulate multiple random matrix. If argument 'dim_in' is
            a list/tuple of integers, then generate multiple random matrixes.
        """
        # Generate matrix W.
        if   self.W is not None         : pass
        elif hasattr(dim_in, "__iter__"): self.W = tuple([self.mat(d) for d in dim_in])
        else                            : self.W = self.mat(dim_in)

        # Generate vector b.
        if   self.b is not None         : pass
        elif hasattr(dim_in, "__iter__"): self.b = tuple([np.random.uniform(0, 2*np.pi, size=self.W.shape[1]) for _ in dim_in])
        else                            : self.b = np.random.uniform(0, 2*np.pi, size=self.W.shape[1])



import sklearn


class RandomFFRegressor(Base):
    """
    Regression with random matrix (RFF/ORF).
    """
    def __init__(self, rand_type, reg_fun=LinearRegression, dim_kernel=16, std_kernel=0.1, W=None, b=None):
        """
        Constractor. Save hyper parameters as member variables and create LinearRegression instance.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            W          (np.ndarray): Random matrix for the input `X`. If None then generated automatically.
            b          (np.ndarray): Random bias for the input `X`. If None then generated automatically.
            args       (dict)      : Extra arguments. This arguments will be passed to the constructor of sklearn's LinearRegression model.
        """
        Base.__init__(self, rand_type, dim_kernel, std_kernel, W, b)
        self.reg = reg_fun()
        self.target_cols = None


    def fit(self, x, y, **args):
        """
        Trains the RFF regression model according to the given data.

        Args:
            X    (pd.DataFrame): Input matrix with shape (n_samples, n_features_input).
            y    (pd.DataFrame): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments. This arguments will be passed to the sklearn's `fit` function.

        Returns:
            (rfflearn.cpu.Regression): Fitted estimator.
        """
        self.set_weight(x.shape[1])
        self.reg.fit(self.conv(x), y, **args)
        self.target_cols = y.columns
        return self

    def predict(self, x, **args):
        """
        Performs prediction on the given data.

        Args:
            x    (pd.DataFrame): Input matrix with shape (n_samples, n_features_input).
            args (dict)      : Extra arguments. This arguments will be passed to the sklearn's `predict` function.

        Returns:
            (np.ndarray): Predicted vector.
        """
        self.set_weight(x.shape[1])
        return pd.DataFrame(self.reg.predict(self.conv(x)), index=x.index, columns=self.target_cols)

    def score(self, X, y, **args):
        """
        Returns the R2 score (coefficient of determination) of the prediction.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments. This arguments will be passed to sklearn's `score` function.

        Returns:
            (float): R2 score of the prediction.
        """
        self.set_weight(X.shape[1])
        return self.reg.score(self.conv(X), y, **args)


# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.


class RFFRegression(RandomFFRegressor, ScenarioGenerator):
    """
    Regression with RFF.
    """
    def __init__(self, dim_kernel=16, std_kernel=0.1, W=None, b=None, q_vect=None, nodes_at_step=None, val_ratio=None,
                 logger=None, n_scen_fit=100, additional_node=False, base_model=LinearRegression, **scengen_kwgs):
        RandomFFRegressor.__init__(self, "rff", base_model, dim_kernel, std_kernel, W, b)
        ScenarioGenerator.__init__(self, q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, logger=logger,
                                   n_scen_fit=n_scen_fit, additional_node=additional_node, **scengen_kwgs)
        self.scaler = None

    def fit(self, x, y, **args):
        self.scaler = sklearn.preprocessing.StandardScaler().fit(x)
        x, y, x_val, y_val = self.train_val_split(x, y)
        x = pd.DataFrame(self.scaler.transform(x), index=x.index)
        RandomFFRegressor.fit(self, x, y, **args)
        ScenarioGenerator.fit(self, x_val, y_val)
        return self

    def predict(self, x, **kwargs):
        x = pd.DataFrame(self.scaler.transform(x), index=x.index)
        y_hat = RandomFFRegressor.predict(self, x, **kwargs)
        y_hat = self.anti_transform(x, y_hat)
        return y_hat
    def predict_quantiles(self, x:pd.DataFrame, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        for h in np.unique(x.index.hour):
            preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        return preds


class AdditiveRFFRegression(ScenarioGenerator):
    """
    Regression with RFF.
    """
    def __init__(self, n_models, dim_kernel=16, std_kernel=0.1, W=None, b=None, q_vect=None, nodes_at_step=None,
                 val_ratio=None, logger=None, n_scen_fit=100, additional_node=False, base_model=LinearRegression, **scengen_kwgs):
        self.n_models = n_models
        self.models = [RandomFFRegressor("rff", base_model, dim_kernel, std_kernel, W, b) for _ in range(n_models)]
        ScenarioGenerator.__init__(self, q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, logger=logger,
                                   n_scen_fit=n_scen_fit, additional_node=additional_node, **scengen_kwgs)
        self.scaler = None

    def fit(self, x, y, **args):
        """Boosting fit."""
        self.scaler = sklearn.preprocessing.StandardScaler().fit(x)
        x, y, x_val, y_val = self.train_val_split(x, y)
        x = pd.DataFrame(self.scaler.transform(x), index=x.index)
        err = y.copy()
        for m in self.models:
            m.fit(x, err, **args)
            y_hat = m.predict(x)
            err = err - y_hat
        ScenarioGenerator.fit(self, x_val, y_val)
        return self

    def predict(self, x, **kwargs):
        """Boosting predict."""
        x = pd.DataFrame(self.scaler.transform(x), index=x.index)
        y_hat = pd.DataFrame(np.zeros((x.shape[0], len(self.models[0].target_cols))), index=x.index, columns=self.target_cols)
        for m in self.models:
            y_hat += m.predict(x)

        y_hat = self.anti_transform(x, y_hat)
        return y_hat
    def predict_quantiles(self, x:pd.DataFrame, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        for h in np.unique(x.index.hour):
            preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        return preds