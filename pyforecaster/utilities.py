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
    x = np.dot(F, x)
    P = np.dot(F, np.dot(P, F.T))  + Q
    return x, P
@njit
def kalman_update(x, P, y, H, R):
    S = np.dot(H, np.dot(P, H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.pinv(S)))
    x = x + np.dot(K, (y - np.dot(H , x)))
    P = P - np.dot(K,np.dot(H, P))
    I_KH = np.eye(K.shape[0]) - np.dot(K, H)
    P = np.dot(I_KH,np.dot(P,I_KH.T)) + np.dot(K, np.dot(R, K.T))
    return x, P
@njit
def kalman(x, P, F, Q, y, H, R):
    x, P = kalman_predict(x, P, F, Q)
    x, P = kalman_update(x, P, y, H, R)
    return x, P


def find_most_important_frequencies(x, n_max=10):
    # Perform FFT on the signal
    fft_x = np.fft.fft(x)

    # Calculate the magnitude spectrum
    mag_spectrum = np.abs(fft_x)[:len(x) // 2]

    # Find the indices corresponding to the highest magnitudes
    return np.sort(np.argsort(mag_spectrum)[::-1][:n_max]) + 1
