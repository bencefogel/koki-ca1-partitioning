import pandas as pd
import numpy as np
from pandas import DataFrame


def load_df(index_fname: str, values_fname: str):
    """
    Loads a DataFrame from a CSV file containing a multiindex and a NumPy file containing the corresponding values.

    Parameters:
        index_fname (str): The file path to the CSV file containing the multiindex data.
        values_fname (str): The file path to the .npy file containing the array of values.

    Returns:
        pd.DataFrame: A pandas DataFrame constructed using the multiindex from the CSV file and the values from the .npy file.
    """
    index = pd.read_csv(index_fname)
    values = np.load(values_fname)

    multiindex = pd.MultiIndex.from_frame(index)
    df = pd.DataFrame(data=values, index=multiindex)
    return df


def get_iax(df: DataFrame, segment: str) -> DataFrame:
    """
    Extracts axial currents associated with a specific segment from a DataFrame.

    Axial currents are calculated as follows:
    - For rows where the segment is a reference node ("ref"), the axial currents are negated.
    - For rows where the segment is a parent node ("par"), the axial currents are taken as-is.

    Parameters:
        df (DataFrame): A multi-index DataFrame containing axial currents.
                        The index must include levels "ref" and "par" for reference and parent nodes.
        segment (str): The name of the segment for which axial currents are extracted.

    Returns:
        DataFrame: A DataFrame containing the axial currents for the specified segment.
    """
    ref_mask = df.index.get_level_values("ref") == segment
    ref_iax = -1 * df[ref_mask]

    par_mask = df.index.get_level_values("par") == segment
    par_iax = df[par_mask]

    df_iax_seg = pd.concat([ref_iax, par_iax], axis=0)

    return df_iax_seg


def plot_sums(im_part_pos: pd.DataFrame, im_part_neg: pd.DataFrame, df_im: pd.DataFrame, df_iax: pd.DataFrame, tps: list, segment: str) -> None:
    """
    Plots the comparison of partitioned and original membrane currents for a specific segment over timepoints.

    This function visualizes the sum of positive and negative currents, ensuring that the partitioned membrane
    currents are consistent with the sum of the original membrane and axial currents. Separate plots are generated
    for positive and negative currents.

    Parameters:
        im_part_pos (pd.DataFrame):
            A DataFrame containing partitioned positive membrane currents.
        im_part_neg (pd.DataFrame):
            A DataFrame containing partitioned negative membrane currents.
        df_im (pd.DataFrame):
            A DataFrame containing the original membrane currents (both positive and negative) for all segments
            and timepoints.
        df_iax (pd.DataFrame):
            A DataFrame containing the original axial currents for all segments and timepoints.
        tps (list):
            A list of timepoints for which the comparison and plotting are performed.
        segment (str):
            The name of the segment for which the currents are analyzed and plotted.

    Returns:
        None: The function generates and displays the plots but does not return any value.

    Notes:
        - Positive currents are those with values >= 0, and negative currents are those with values < 0.
    """
    df_im_part_calculated_pos = im_part_pos
    df_im_part_calculated_neg = im_part_neg
    df_im_part_calculated_pos_sum = df_im_part_calculated_pos.sum(axis=0).values
    df_im_part_calculated_neg_sum = df_im_part_calculated_neg.sum(axis=0).values

    df_im_original = df_im.iloc[:, tps].loc[segment]
    df_im_original_pos = df_im_original[df_im_original >= 0].fillna(0)
    df_im_original_neg = df_im_original[df_im_original < 0].fillna(0)
    df_im_original_pos_sum = df_im_original_pos.sum(axis=0).values
    df_im_original_neg_sum = df_im_original_neg.sum(axis=0).values

    df_iax_original = get_iax(df_iax, segment).iloc[:, tps]
    df_iax_original_pos = df_iax_original[df_iax_original >= 0].fillna(0)
    df_iax_original_neg = df_iax_original[df_iax_original < 0].fillna(0)
    df_iax_original_pos_sum = df_iax_original_pos.sum(axis=0).values
    df_iax_original_neg_sum = df_iax_original_neg.sum(axis=0).values

    # Plotting
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Positive values
    axes[0].plot(df_im_part_calculated_pos_sum, label="partitioned", marker='o')
    axes[0].plot(df_im_original_pos_sum + df_iax_original_pos_sum, label="membrane + axial", marker='x')
    axes[0].set_title("positive currents")
    axes[0].set_xlabel("timepoints")
    axes[0].set_ylabel("sum of positive currents")
    axes[0].legend()

    # Negative values
    axes[1].plot(df_im_part_calculated_neg_sum, label="partitioned", marker='o')
    axes[1].plot(df_im_original_neg_sum + df_iax_original_neg_sum, label="membrane + axial", marker='x')
    axes[1].set_title("negative currents")
    axes[1].set_xlabel("timepoints")
    axes[1].legend()

    plt.tight_layout()
    plt.show()