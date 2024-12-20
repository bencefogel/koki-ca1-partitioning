import pandas as pd
import numpy as np
from pandas import DataFrame


def load_df(index_fname, values_fname):
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
