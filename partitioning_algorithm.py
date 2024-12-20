import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from tqdm import tqdm
from partitioning_order import create_directed_graph, get_partitioning_order


def partition_iax(im: DataFrame, iax: DataFrame, timepoints: list, target: str) -> DataFrame:
    """
    Partitions axial currents into membrane currents across multiple time points and updates a copy of the membrane currents DataFrame.

    Parameters:
        im (DataFrame): A DataFrame containing membrane currents for each node and time point.
        iax (DataFrame): A DataFrame containing axial currents for each reference-parent pair and time point.
        timepoints (list): A list of time points for which the partitioning process is performed.
        target (str): The target node to start the partitioning traversal from.

    Returns:
        DataFrame: A modified copy of `im` with updated membrane currents after partitioning axial currents for all time points.
    """
    im_part = im.copy()

    for tp in tqdm(timepoints):
        dg = create_directed_graph(iax, tp)

        partitioning_order_out = get_partitioning_order(dg, target, 'out')
        for segment_pair in partitioning_order_out:
            ref = segment_pair[0]
            par = segment_pair[1]
            partition_iax_single(ref, par, tp, im_part, iax)

        partitioning_order_in = get_partitioning_order(dg, target, 'in')
        for segment_pair in partitioning_order_in:
            ref = segment_pair[0]
            par = segment_pair[1]
            partition_iax_single(ref, par, tp, im_part, iax)

    return im_part


def partition_iax_single(ref: str, par: str, tp: int, im: DataFrame, iax: DataFrame) -> None:
    """
    Partitions axial currents at a specific time point into membrane currents and updates the parent node's membrane currents.

    Parameters:
        ref (str): The reference node (child node) in the current partitioning process.
        par (str): The parent node in the current partitioning process.
        tp (int): The time point for which the partitioning is performed.
        im (DataFrame): A DataFrame containing membrane currents for each node and time point.
        iax (DataFrame): A DataFrame containing axial currents for each reference-parent pair and time point.

    Returns:
        None: The function updates the `df_im` DataFrame in place with the partitioned axial currents added to the parent node's membrane currents.
    """
    im_tp = im.loc[ref, tp]
    im_tp_out = im_tp[im_tp >= 0]
    im_tp_in = im_tp[im_tp < 0]
    iax_tp = iax.loc[(ref, par), tp]

    # Calculate partitioned membrane currents
    if iax_tp >= 0:
        part_curr = get_part_out(iax_tp, im_tp_out)
    elif iax_tp < 0:
        part_curr = get_part_in(iax_tp, im_tp_in)

    # Update membrane currents of the parent with the partitioned axial current values
    part_curr = part_curr.reindex(im.loc[par, tp].index, fill_value=0)
    updated_curr = im.loc[par, tp] + part_curr
    im.loc[par, tp] = updated_curr.values.astype(np.float32)  # update original dataframe of the membrane currents


def get_part_out(iax_tp: float, im_tp_out: Series) -> Series:
    """
    Partitions a positive axial current into outward membrane currents.

    Parameters:
        iax_tp (float): The axial current value to be partitioned (must be positive).
        im_tp_out (pd.Series): A pandas Series containing outward membrane currents (all values >= 0).

    Returns:
        pd.Series: A pandas Series containing the partitioned axial current components.
    """
    assert (im_tp_out >= 0).all()
    ratios = im_tp_out / im_tp_out.sum()
    partitioned_out = pd.Series(ratios * iax_tp, index=im_tp_out.index)
    return partitioned_out


def get_part_in(iax_tp: float, im_tp_in: Series) -> Series:
    """
    Partitions a negative axial current into inward membrane currents.

    Parameters:
        iax_tp (float): The axial current value to be partitioned (must be negative).
        im_tp_in (pd.Series): A pandas Series containing inward membrane currents (all values < 0).

    Returns:
        pd.Series: A pandas Series containing the partitioned axial current components
    """
    assert (im_tp_in < 0).all()
    ratios = im_tp_in / im_tp_in.sum()
    partitioned_in = pd.Series(ratios * iax_tp, index=im_tp_in.index)
    return partitioned_in



