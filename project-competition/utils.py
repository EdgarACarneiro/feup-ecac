import pandas as pd
import numpy as np
import os


def read_csv_to_df(filename, delimiter=','):
    '''Read a csv to a pandas dataframe'''
    return pd.read_csv(filename, delimiter)


def write_df_to_csv(df, directory, filename):
    '''Write the content of a dataframe to a csv file'''
    csv = df.to_csv(index=False)

    os.makedirs(directory, exist_ok=True)
    output_path = '%s/%s' % (directory, filename)

    with open(output_path, 'w') as fd:
        fd.write(csv)
