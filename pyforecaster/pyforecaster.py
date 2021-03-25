import numpy as np
import pandas as pd


def format_dataset(x):
    assert isinstance(x, pd.DataFrame), 'x must be an instance of pandas.DataFrame'
    return x
