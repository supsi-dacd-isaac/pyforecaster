import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import ray
from multiprocessing import Pool, cpu_count
import sharedmem
from time import time
from typing import Union


def mapper(f, pars, *argv, sharedmem=False, **kwarg):
    # create a shared object from X
    if sharedmem:
        argv = [sharedmem.copy(x) for x in argv]

    # parallel process over shared object
    pool = Pool(cpu_count() - 1)
    res = pool.starmap_async(f, [(p, *argv, *list(kwarg.values())) for p in pars])
    a = res.get()
    pool.close()
    pool.join()
    return a


def get_logger(level=logging.INFO):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s')
    logger.setLevel(level)
    return logger


def array_split(df, n_splits, axis=0):
    """
    Do what np.array_split does, but without changing dtype (why in the world np.array_split does it?)
    :param df:
    :param n_splits:
    :param axis:
    :return:
    """
    is_df = isinstance(df, pd.DataFrame)
    split_points = np.linspace(0, df.shape[axis], n_splits+1).astype(int)
    dfs = []
    for s in range(n_splits):
        idxs = np.arange(split_points[s],split_points[s+1])
        if axis == 1:
            fold = df.iloc[:, idxs] if is_df else df[:, idxs]
        else:
            fold = df.iloc[idxs, :] if is_df else df[idxs, :]
        dfs.append(fold)
    return dfs


def fdf_parallel(f, df:Union[pd.DataFrame, list], n_splits=None, axis=0, use_ray=True):
    """
    Parallelize a function to be applied on a pd.DataFrame or on a list of pd.DataFrames.
    :param f:
    :param df: pd.DataFrame to be splitted or a list of pd.DataFrame already separated
    :param n_splits:
    :param axis: parallelization axis. If 0, df is splitted in horizontal slices, if 1 df is splitted vertically
    :return:
    """

    assert axis in [0, 1], 'axis must be in {0, 1}'

    if n_splits is None:
        n_splits = cpu_count()

    # if a pd.DataFrame was passed, obtain df splits
    if isinstance(df, pd.DataFrame):
        if df.shape[1] > n_splits:
            dfs = array_split(df, n_splits, axis=axis)
        else:
            dfs = [df]
    elif isinstance(df, list):
        dfs = df
    else:
        raise TypeError('df must be a list or a pd.DataFrame')

    test_df = dfs[0].iloc[:n_splits, :] if axis == 0 else dfs[0].iloc[:, :n_splits]
    try:
        f(test_df.values)
        numpy_parallelizable = True
    except:
        numpy_parallelizable = False

    if numpy_parallelizable and not use_ray:
        if axis==0:
            df_shape_0 = sum([x.shape[0] for x in dfs])
            df_shape_1 = dfs[0].shape[1]
            df_columns = np.unique(dfs[0].columns)
        else:
            df_shape_0 = dfs[0].shape[0]
            df_shape_1 = sum([x.shape[1] for x in dfs])
            df_columns = np.unique([x.columns for x in dfs])

        df_shape = (df_shape_0, df_shape_1)

        df_indexes = np.hstack([x.index.ravel() for x in dfs]) if axis==0 else dfs[0].index

        res = mapper(f, [df.values for df in [dfs[0]]])
        res = np.vstack(res) if axis == 0 else np.hstack(res)

        if res.shape == df_shape:
            res = pd.DataFrame(res, columns=df_columns, index=df_indexes)

    else:
        if not ray.is_initialized():
            ray.init()
        @ray.remote
        def f_rem(df):
            df = f(df)
            return df
        responses = ray.get([f_rem.remote(df_i) for df_i in dfs])
        if isinstance(responses[0], pd.DataFrame):
            res = pd.concat(responses, axis=axis)
        elif isinstance(responses[0], tuple):
            # this assumes all the outputs of the function are pd.DataFrames
            res = []
            for i in range(len(responses[0])):
                res.append(pd.concat([r[i] for r in responses], axis=axis))
            res = tuple(res)
        else:
            res = np.vstack(responses) if axis == 0 else np.hstack(responses)
        ray.shutdown()

    return res


def reduce_mem_usage_series(s):
    col_type = s.dtype
    if col_type != object:
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
        elif str(col_type)[:5] == 'float':
            # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            #     s = s.astype(np.float16)
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                s = s.astype(np.float32)
            else:
                s = s.astype(np.float64)
    return s


def reduce_mem_usage_df(df):
    df = pd.concat([reduce_mem_usage_series(df[c]) for c in df.columns], axis=1)
    return df


def reduce_mem_usage_np(x):
    try:
        x = np.hstack([reduce_mem_usage_series(x[:, [c]]) for c in range(x.shape[1])])
    except:
        print(1)

    return x


def reduce_mem_usage(df, parallel=True, logger=None, use_ray=True):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    logger = logger if logger is not None else get_logger()

    start_mem = df.memory_usage().sum() / 1024 ** 2
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    if parallel:
        if use_ray:
            df = fdf_parallel(reduce_mem_usage_df, df, axis=1, use_ray=use_ray)
        else:
            df = fdf_parallel(reduce_mem_usage_np, df, axis=1, use_ray=use_ray)
    else:
        for col in tqdm(df.columns):
            df[col] = reduce_mem_usage_series(df[col])

    end_mem = df.memory_usage().sum() / 1024 ** 2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def f_test(x):
    return x/x.mean()


if __name__ == '__main__':
    logger = get_logger()
    N = 100
    t = 10000
    df = pd.DataFrame(np.random.randn(t, N)**4)
    """
    res = fdf_parallel(f_test, df)
    

    t0 = time()
    dfa = reduce_mem_usage(df, logger=logger, parallel=False)
    print('time without parallelizing: {}'.format(time() - t0))
    """

    t0 = time()
    dfa = reduce_mem_usage(df, logger=logger, parallel=True, use_ray=True)
    print('time using ray: {}'.format(time() - t0))

    t0 = time()
    dfa = reduce_mem_usage(df, logger=logger, parallel=True, use_ray=False)
    print('time using mp: {}'.format(time() - t0))
