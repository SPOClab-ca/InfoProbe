from functools import wraps
import os, time, sys
import pandas as pd
import numpy as np


def timed_func(foo):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        results = foo(*args, **kwargs)
        print ("{} done in {:.2f} seconds.".format(foo.__name__, time.time() - start_time))
        return results
    return wrapper


def combine_df(results_dir, verbose=False):
    comb = None
    for fn in os.listdir(results_dir):
        if not fn.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(results_dir, fn))
        if comb is None:
            comb = df
        else:
            comb = pd.concat([comb, df], ignore_index=True)
            
    if verbose:
        print ("Combined df. Total len={}".format(len(comb)))
    return comb


def groupby_return_df(df, group_features, task_features):
    dfs = {}
    for ft in group_features:
        dfs[ft] = []
    for ft in task_features:
        dfs[ft] = []
    df_with_mean = df.groupby(group_features).mean()
    for idx, row in df_with_mean.iterrows():
        for j, ft in enumerate(group_features):
            dfs[ft].append(idx[j])
        for j, ft in enumerate(task_features):
            dfs[ft].append(row[ft])
    return pd.DataFrame(dfs)
