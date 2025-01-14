from concurrent.futures import ProcessPoolExecutor
import glob
import os
from utils import load_df, get_soma_currents_dataframes, get_itotal_dataframes
from partitioning_algorithm import partition_iax

input_dir = 'E:/cluster_seed30/preprocessed_data'
output_dir = 'E:/cluster_seed30/partitioned_data'

# Ensure the output directories exist
partitioned_dir = os.path.join(output_dir, 'partitioned_currents')
itotal_dir = os.path.join(output_dir, 'total_currents')
soma_dir = os.path.join(output_dir, 'soma_currents')

os.makedirs(partitioned_dir, exist_ok=True)
os.makedirs(itotal_dir, exist_ok=True)
os.makedirs(soma_dir, exist_ok=True)

iax_index_file = input_dir + '/axial_currents_merged_soma/multiindex_merged_soma.csv'
iax_values_files = glob.glob(input_dir + '/axial_currents_merged_soma/*.npy')

im_index_file = input_dir + '/membrane_currents_merged_soma/multiindex_merged_soma.csv'
im_values_files = glob.glob(input_dir + '/membrane_currents_merged_soma/*.npy')

segment = 'soma'
batch_size = 1 #  number of files to process parallel


def extract_file_number(filepath):
    """Extract the number from the file name."""
    filename = os.path.basename(filepath)
    number = ''.join(filter(str.isdigit, filename))
    return number


def process_file_pair(im_file, iax_file):
    # Log the files being processed
    print(f"Reading and processing files: im = {im_file}, iax = {iax_file}")
    # Load data for the given pair of files
    df_iax = load_df(iax_index_file, iax_file)
    df_im = load_df(im_index_file, im_file)

    # Perform the partitioning algorithm
    im_part_pos, im_part_neg = partition_iax(df_im, df_iax, timepoints=list(range(50,52)), target=segment)
    # to test on smaller data: timepoints=list(range(50,52))
    # process full dataset: timepoints=df_iax.columns

    # Extract file numbers
    im_number = extract_file_number(im_file)

    # Save partitioned currents
    im_pos_output_path = os.path.join(partitioned_dir, f"im_part_pos_{im_number}.csv")
    im_neg_output_path = os.path.join(partitioned_dir, f"im_part_neg_{im_number}.csv")
    im_part_pos.to_csv(im_pos_output_path, index=True)
    im_part_neg.to_csv(im_neg_output_path, index=True)

    # Process soma currents
    soma_pos, soma_neg = get_soma_currents_dataframes(df_im, df_iax)
    soma_pos_path = os.path.join(soma_dir, f"soma_pos_{im_number}.csv")
    soma_neg_path = os.path.join(soma_dir, f"soma_neg_{im_number}.csv")
    soma_pos.to_csv(soma_pos_path, index=True)
    soma_neg.to_csv(soma_neg_path, index=True)

    # Calculate itotal
    itotal_pos, itotal_neg = get_itotal_dataframes(df_im)
    itotal_pos_path = os.path.join(itotal_dir, f"itotal_pos_{im_number}.csv")
    itotal_neg_path = os.path.join(itotal_dir, f"itotal_neg_{im_number}.csv")
    itotal_pos.to_csv(itotal_pos_path, index=True)
    itotal_neg.to_csv(itotal_neg_path, index=True)


def process_in_batches(batch_size=1):
    """Process files in batches to avoid overloading system resources."""
    # Split the file pairs into smaller batches
    batches = [list(zip(im_values_files[i:i + batch_size], iax_values_files[i:i + batch_size]))
               for i in range(0, len(im_values_files), batch_size)]

    # Iterate over each batch and process it
    for batch in batches:
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_file_pair, im_file, iax_file) for im_file, iax_file in batch]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing files: {e}")


if __name__ == '__main__':
    process_in_batches(batch_size=batch_size)
