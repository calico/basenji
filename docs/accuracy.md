### Basenji
###### Sequential regulatory activity predictions with deep convolutional neural networks.
--------------------------------------------------------------------------------
## Accuracy

<a name="test"/>

### basenji_test.py

Compute accuracy statistics for a trained model on a held out test set.

| Arguments | Type | Description |
| --- | --- | --- |
| params_file | Text table | Model configuration parameters. |
| model file | TensorFlow Saver file | Trained Basenji model. |
| test_hdf5_file | HDF5 | Output HDF5 file with test_in/test_out keys. |

--------------------------------------------------------------------------------
<a name="test_genes"/>

### basenji_test_genes.py

Compute accuracy statistics for a trained model at gene TSSs.

| Arguments | Type | Description |
| --- | --- | --- |
| params_file | Text table | Model configuration parameters. |
| model file | TensorFlow Saver file | Trained Basenji model. |
| genes_hdf5_file | HDF5 | HDF5 file with gene sequences and descriptions. |
