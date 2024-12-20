import pandas as pd
import numpy as np


def load_df(index_fname, values_fname):
    index = pd.read_csv(index_fname)
    values = np.load(values_fname)

    multiindex = pd.MultiIndex.from_frame(index)
    df = pd.DataFrame(data=values, index=multiindex)
    return df
