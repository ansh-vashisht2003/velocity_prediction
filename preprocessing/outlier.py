import pandas as pd

def remove_outliers(df):

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    IQR = Q3 - Q1

    df = df[~((df < (Q1 - 1.5 * IQR)) |
              (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df