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
tps = list(df_iax.columns)[0:500]
segment = 'soma'

df_im_part = partition_iax(df_im, df_iax, timepoints=tps, target=segment)

# Checking results
from utils import get_iax
df_im_part_calculated = df_im_part.iloc[:, tps].loc[segment]
df_im_part_calculated_pos = df_im_part_calculated[df_im_part_calculated >= 0].fillna(0)
df_im_part_calculated_neg = df_im_part_calculated[df_im_part_calculated < 0].fillna(0)
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
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)

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
axes[1].set_title("positive currents")
axes[1].set_xlabel("timepoints")
axes[1].legend()

plt.tight_layout()
plt.show()


