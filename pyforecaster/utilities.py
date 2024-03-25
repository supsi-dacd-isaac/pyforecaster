import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

def from_tensor_to_pdmultiindex(x, index, first_index, second_index):
    df = {}
    for i, si in zip(range(x.shape[2]), second_index):
       df[si] = pd.DataFrame(x[:, :, i], index=index, columns=first_index)
    return pd.concat(df, axis=1)

def convert_multiindex_pandas_to_tensor(df, zero_first=True):
    if df.columns.nlevels < 2:
        return df
    c_shape = df.columns.levshape
    if zero_first:
        return df.values.reshape(-1, c_shape[0], c_shape[1])
    else:
        return np.swapaxes(df.values.reshape(-1, c_shape[0], c_shape[1]), 1, 2)

def get_logger(level=logging.INFO, name='pyforecaster'):
    logger = logging.getLogger(name)
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s')
    logger.setLevel(level)
    return logger

@njit
def kalman_predict(x, P, F, Q):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P
@njit
def kalman_update(x, P, y, H, R):
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.pinv(S)
    x = x + K @ (y - H @ x)
    P = P - K @ H @ P
    I_KH = np.eye(K.shape[0]) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    return x, P
@njit
def kalman(x, P, F, Q, y, H, R):
    x, P = kalman_predict(x, P, F, Q)
    x, P = kalman_update(x, P, y, H, R)
    return x, P

