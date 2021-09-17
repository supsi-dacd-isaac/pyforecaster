import numpy as np

def feature_importance(x, m, n_epsilon=10):
    # isotropic spherical sampling
    n_o, n_f = x.shape()
    eps = np.random.randn(n_epsilon, n_o, n_f)
    preds = np.dstack([m.predict((x + eps_i)) for eps_i in eps])
    std = np.std(preds, axis=-1)
    

def a_priori_distance(x, metric='euclidean'):
    pass

def a_posteriori_distance(x, m, metric='euclidean'):
    pass