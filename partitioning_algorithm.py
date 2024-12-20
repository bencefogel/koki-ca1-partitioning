import pandas as pd
import numpy as np
from pandas import DataFrame, Series


def partition_iax(ref: str, par: str, tp: int, df_im: DataFrame, df_iax: DataFrame) -> None:
    im_tp = df_im.loc[ref, tp]
    im_tp_out = im_tp[im_tp >= 0]
    im_tp_in = im_tp[im_tp < 0]
    iax_tp = df_iax.loc[(ref, par), tp]

    # Calculate partitioned membrane currents
    if iax_tp >= 0:
        part_curr = get_part_out(iax_tp, im_tp_out)
    elif iax_tp < 0:
        part_curr = get_part_in(iax_tp, im_tp_in)

    # Update membrane currents of the parent with the partitioned axial current values
    part_curr = part_curr.reindex(df_im.loc[par, tp].index, fill_value=0)
    updated_curr = df_im.loc[par, tp] + part_curr
    df_im.loc[par, tp] = updated_curr.values.astype(np.float32)  # update original dataframe of the membrane currents


def get_part_out(iax_tp: float, im_tp_out: Series) -> Series:
    assert (im_tp_out >= 0).all()
    ratios = im_tp_out / im_tp_out.sum()
    partitioned_out = pd.Series(ratios * iax_tp, index=im_tp_out.index)
    return partitioned_out


def get_part_in(iax_tp: float, im_tp_in: Series) -> Series:
    assert (im_tp_in < 0).all()
    ratios = im_tp_in / im_tp_in.sum()
    partitioned_in = pd.Series(ratios * iax_tp, index=im_tp_in.index)
    return partitioned_in



