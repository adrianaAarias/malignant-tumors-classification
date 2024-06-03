import pandas as pd
import numpy as np


def count_non_numeric_values(series_col, verbose = True):
    '''Description: This function takes a series column to compute the count and ratio of nonfloat values.
        ARGS
            - series_col (pandas series): pandas series column
        RETURNS
            - count (integer): Number of nonfloat rows
            - ratio (float): Number of nonfloat rows over total rows
    '''

    series_formatted = pd.to_numeric(series_col, errors = 'coerce')
    count = pd.isnull(series_formatted).sum()
    ratio = count/len(series_formatted)
    if verbose:
        print(f"In the column {series_col.name} There are {count} non numeric values which represent a ratio of {round(ratio*100, 2)} %")
    return count, ratio


def display_non_numeric_values(series_col):
    count,_ = count_non_numeric_values(series_col, verbose = False)
    if count>0:
        series_formatted = pd.to_numeric(series_col, errors = 'coerce') 
        print(50*"*")
        print(50*"*")
        print(f"Displaying non numeric values from the column {series_col.name}")
        display(series_col[pd.isnull(series_formatted)])
    return


def replace_non_numeric_values(series_col):
    series_formatted = pd.to_numeric(series_col, errors = 'coerce')
    return series_formatted


def convert_to_numeric(series_col):
    series_formatted = pd.to_numeric(series_col, errors = 'coerce')
    series_col[~pd.isnull(series_formatted)] = series_formatted[~pd.isnull(series_formatted)] 
    return series_col


def outlier_elimination(series, q_min = 0.25, q_max = 0.75, threshold = 3):
    quartiles = series.quantile([q_min,q_max])
    iqr = quartiles[q_max] - quartiles[q_min]
    lower_bound = quartiles[q_min] - threshold*iqr
    upper_bound = quartiles[q_max] + threshold*iqr
    filter_condition = (series >= lower_bound) & (series <= upper_bound)
    series[~filter_condition] = np.NaN
    return series