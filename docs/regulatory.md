### Basenji
###### Sequential regulatory activity predictions with deep convolutional neural networks.
--------------------------------------------------------------------------------
## Regulatory element analysis

<a name="motifs"/>

### basenji_motifs.py

Perform motif analysis on sequences recognized by the first convolution layer filters.

| Arguments | Type | Description |
| --- | --- | --- |
| params_file | Text table | Model configuration parameters. |
| model file | TensorFlow Saver file | Trained Basenji model. |
| data_file | HDF5 | HDF5 file with test_in/test_out keys. |

--------------------------------------------------------------------------------
<a name="sat"/>

### basenji_sat.py

Perform an in silico saturated mutagenesis of the given test sequences.

| Arguments | Type | Description |
| --- | --- | --- |
| params_file | Text table | Model configuration parameters. |
| model file | TensorFlow Saver file | Trained Basenji model. |
| input_file | HDF5/FASTA | HDF5 file with test_in/test_out keys, or FASTA file of sequences. |

--------------------------------------------------------------------------------
<a name="map"/>

### basenji_map.py

Visualize a sequence's prediction's gradients as a map of influence across the genomic region.

| Arguments | Type | Description |
| --- | --- | --- |
| params_file | Text table | Model configuration parameters. |
| model file | TensorFlow Saver file | Trained Basenji model. |
| genes_hdf5_file | HDF5 | HDF5 file with gene sequences and descriptions. |