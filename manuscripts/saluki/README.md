## Saluki: The genetic and biochemical determinants of mRNA degradation rates in mammals
--------------------------------------------------------------------------------

### Manuscript model and data

[The genetic and biochemical determinants of mRNA degradation rates in mammals. bioRxiv 4/2022.]()

A reproduction of each figure in the paper, along with associated code to generate predictions, can be found [here](https://github.com/vagarwal87/saluki_paper).

All associated saved models as well as training, validation, and test TFRecords files can be found here:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6326409.svg)](https://doi.org/10.5281/zenodo.6326409)

in the datasets/deeplearning/train_gru subdirectory.

All files in this [project](https://github.com/calico/basenji/tree/master/bin) directory, listed under saluki*.py, are also associated with this work.

#### Note about *in silico* mutagensis (ISM) scores: 
We mean-centered the ISM scores as a normalization for the four nucleotides at each position such that they sum to zero. The normalized ISM score at the reference nucleotide can be used as an importance score for that position.
