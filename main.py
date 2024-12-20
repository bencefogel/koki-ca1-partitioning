import numpy as np
import pandas as pd

from utils import load_df
from partitioning_algorithm import partition_iax


input_dir = 'L:/cluster_seed30/preprocessed_data'

iax_index_file = input_dir + '/axial_currents_merged_soma/multiindex_merged_soma.csv'
iax_values_file = input_dir + '/axial_currents_merged_soma/merged_soma_values_4.npy'

im_index_file = input_dir + '/membrane_currents_merged_soma/multiindex_merged_soma.csv'
im_values_file = input_dir + '/membrane_currents_merged_soma/merged_soma_values_4.npy'

df_iax = load_df(iax_index_file, iax_values_file)
df_im = load_df(im_index_file, im_values_file)
tps = list(df_iax.columns)
segment = 'soma'

df_im_part = partition_iax(df_im, df_iax, timepoints=tps, target=soma)
