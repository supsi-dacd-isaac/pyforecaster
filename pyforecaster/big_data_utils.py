import pandas as pd
import numpy as np
from tqdm import tqdm
from pandarallel import pandarallel
import logging
import ray


def get_logger(level=logging.INFO):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s')
    logger.setLevel(level)
    return logger


def parallelize_columns(f, df, n_splits=None):
    if not ray.is_initialized():
        ray.init()
    if n_splits is None:
        n_splits = int(ray.available_resources()['CPU'])

    @ray.remote
    def f_rem(df):
        for c in df.columns:
            df[c] = f(df[c])
        return df

    dfs = np.array_split(df, n_splits, axis=1)
    res = pd.concat(ray.get([f_rem.remote(df_i) for df_i in dfs]), axis=1)
    ray.shutdown()
    return res

# This function is used to reduce memory of a pandas dataframe
# The idea is cast the numeric type to another more memory-effective type
# For ex: Features "age" should only need type='np.int8'
# Source: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage


def reduce_mem_usage_series(s):
    col_type = s.dtype

    if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
        c_min = s.min()
        c_max = s.max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                s = s.astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                s = s.astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                s = s.astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                s = s.astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                s = s.astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                s = s.astype(np.float32)
            else:
                s = s.astype(np.float64)
    elif 'datetime' not in col_type.name:
        s = s.astype('category')
    return s


def reduce_mem_usage(df, parallel=True, logger=None):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    logger = logger if logger is not None else get_logger()

    start_mem = df.memory_usage().sum() / 1024 ** 2
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    if parallel:
        df = parallelize_columns(reduce_mem_usage_series, df)
    else:
        for col in tqdm(df.columns):
            df[col] = reduce_mem_usage_series(df[col])

    end_mem = df.memory_usage().sum() / 1024 ** 2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


if __name__ == '__main__':
    logger = get_logger()
    N = 1000
    t = 10000
    df = pd.DataFrame(np.random.randn(t, N))
    print(df.memory_usage().sum())
    dfa = reduce_mem_usage(df, logger=logger, parallel=True)
    print(dfa.memory_usage().sum())


