### Basenji
###### Sequential regulatory activity predictions with deep convolutional neural networks.
--------------------------------------------------------------------------------
## Preprocess

<a name="bam_cov.py"/>
### bam_cov.py

Transform BAM alignments to a normalized BigWig (or HDF5-stored) coverage track.

| Arguments | Type | Description |
| --- | --- | --- |
| bam_file | BAM/CRAM | Alignments file from which to extract coverage. |
| output_file | BigWig/HDF5 | Output coverage track stored as BigWig or HDF5, depending on the ".bw" suffix. |
| hdf5_file | HDF5 | Output HDF5 file with train_in/train_out, test_in/test_out and many other keys. |

--------------------------------------------------------------------------------
<a name="basenji_hdf5_single.py"/>
### basenji_hdf5_single.py

Combine a set of coverage tracks stored as BigWig or HDF5 into a single file for training and testing, parallelizing over samples per-segment using multiprocessing on a single machine.

| Arguments | Type | Description |
| --- | --- | --- |
| fasta_file | FASTA | FASTA file of chromosome sequences. |
| sample_wigs_file | Text table | Sample labels and paths to coverage files. |
| hdf5_file | HDF5 | Output HDF5 file with train_in/train_out, test_in/test_out and many other keys. |

--------------------------------------------------------------------------------
<a name="basenji_hdf5_cluster.py"/>
### basenji_hdf5_cluster.py

Combine a set of coverage tracks stored as BigWig or HDF5 into a single file for training and testing, parallelizing over samples on our SLURM cluster.

| Arguments | Type | Description |
| --- | --- | --- |
| fasta_file | FASTA | FASTA file of chromosome sequences. |
| sample_wigs_file | Text table | Sample labels and paths to coverage files. |
| hdf5_file | HDF5 | Output HDF5 file with train_in/train_out, test_in/test_out and many other keys. |

--------------------------------------------------------------------------------
<a name="basenji_genes.py"/>
### basenji_genes.py

Tile a set of genes and save the result in HDF5 for Basenji processing.

| Arguments | Type | Description |
| --- | --- | --- |
| fasta_file | FASTA | FASTA file of chromosome sequences. |
| gtf_file | GTF | Gene annotations in gene transfer format. |
| hdf5_file | HDF5 | Output HDF5 file with gene sequences and descriptions. |