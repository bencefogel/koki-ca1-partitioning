import os
import pandas as pd
from utils import load_df, plot_sums
from partitioning_algorithm import partition_iax


# Set parameters
input_dir = 'E:/cluster_seed30/preprocessed_data/'
output_dir = 'E:/cluster_seed30/partitioned_data/'
segment = 'soma'
chunk_size = 10000

# Load DataFrames
df_iax = load_df(input_dir+'axial_currents_merged_soma/multiindex_merged_soma.csv', input_dir+'axial_currents_merged_soma/merged_soma_values_0.npy')
df_im = load_df(input_dir+'membrane_currents_merged_soma/multiindex_merged_soma.csv', input_dir+'membrane_currents_merged_soma/merged_soma_values_0.npy')

# Initialize empty lists for partitioned results
im_part_pos_list = []
im_part_neg_list = []

# Process DataFrame in column chunks
num_columns = len(df_iax.columns)
for start_col in range(0, num_columns, chunk_size):
    # Define the column range for this chunk
    end_col = min(start_col + chunk_size, num_columns)
    col_subset = df_iax.columns[start_col:end_col]

    # Subset DataFrames for the current columns
    df_iax_chunk = df_iax[col_subset]
    df_im_chunk = df_im[col_subset]

    # Partition currents for the current column subset
    chunk_pos, chunk_neg = partition_iax(df_im_chunk, df_iax_chunk, timepoints=list(col_subset), target=segment)

    # Append chunk results to lists
    im_part_pos_list.append(chunk_pos)
    im_part_neg_list.append(chunk_neg)

# Concatenate all chunk results into final DataFrames
im_part_pos = pd.concat(im_part_pos_list, axis=1)
im_part_neg = pd.concat(im_part_neg_list, axis=1)

# Generate output file names with the file_index appended to avoid overwriting
pos_output_file = os.path.join(output_dir, f'partitioned_pos.csv')
neg_output_file = os.path.join(output_dir, f'partitioned_neg.csv')

# Save DataFrames to CSV
im_part_pos.to_csv(pos_output_file, index=False)
im_part_neg.to_csv(neg_output_file, index=False)
