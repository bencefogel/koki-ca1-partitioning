# Partitioning of the axial current into membrane current components
Code for decomposing the axial currents of a CA1 pyramidal neuron model into membrane current (intrinsic and synaptic) components.

## **Features**

This script calculates the partitioned axial current components and returns them in a dataframe.

## **Configuration and Parameters**

To calculate the partitioned axial current components, you need to configure the following parameters in the main script (`main.py`):

1. Set the path to the preprocessed axial and membrane current values in `input_dir`.
2. Set the timepoints that should be processed in `tps`.
**Important note**: Currently the partitioning only works for the somatic compartment!

## **Performance**
- Execution time: Approximately 5 seconds per iteration (where one iteration corresponds to processing a single timepoint).
- This timing reflects the algorithm's performance on DataFrames with the following dimensions:
  - iax: ~1400 x 20k
  - im: ~20k x 20k
  Processing is notably faster for smaller DataFrames.