# MOST OF THIS CODE IS TAKEN FROM https://github.com/tiskw/random-fourier-features/blob/main/rfflearn/cpu/rfflearn_cpu_regression.py
# Author: Tetsuya Ishikawa

import functools
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression

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

    def conv(self, X, index=None):
        """
        Applies random matrix to the given input vectors `X` and create feature vectors.

        Args:
            X     (np.ndarray): Input matrix with shape (n_samples, n_features).
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
        return np.cos(X @ W + b)

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


class Regression(Base):
    """
    Regression with random matrix (RFF/ORF).
    """
    def __init__(self, rand_type, dim_kernel=16, std_kernel=0.1, W=None, b=None, **args):
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
        super().__init__(rand_type, dim_kernel, std_kernel, W, b)
        self.reg = LinearRegression(**args)

    def fit(self, X, y, **args):
        """
        Trains the RFF regression model according to the given data.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments. This arguments will be passed to the sklearn's `fit` function.

        Returns:
            (rfflearn.cpu.Regression): Fitted estimator.
        """
        self.set_weight(X.shape[1])
        self.reg.fit(self.conv(X), y, **args)
        return self

    def predict(self, X, **args):
        """
        Performs prediction on the given data.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            args (dict)      : Extra arguments. This arguments will be passed to the sklearn's `predict` function.

        Returns:
            (np.ndarray): Predicted vector.
        """
        self.set_weight(X.shape[1])
        return self.reg.predict(self.conv(X), **args)

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


class RFFRegression(Regression):
    """
    Regression with RFF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


class ORFRegression(Regression):
    """
    Regression with ORF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)



