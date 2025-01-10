from utils import load_df, plot_sums
from partitioning_algorithm import partition_iax


input_dir = 'E:/cluster_seed30/preprocessed_data'

iax_index_file = input_dir + '/axial_currents_merged_soma/multiindex_merged_soma.csv'
iax_values_file = input_dir + '/axial_currents_merged_soma/merged_soma_values_4.npy'

im_index_file = input_dir + '/membrane_currents_merged_soma/multiindex_merged_soma.csv'
im_values_file = input_dir + '/membrane_currents_merged_soma/merged_soma_values_4.npy'

df_iax = load_df(iax_index_file, iax_values_file)
df_im = load_df(im_index_file, im_values_file)
tps = list(df_iax.columns)[50:60]
segment = 'soma'

im_part_pos, im_part_neg = partition_iax(df_im, df_iax, timepoints=tps, target=segment)

# plot_sums(im_part_pos, im_part_neg, df_im, df_iax, tps, segment)



