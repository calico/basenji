### Basenji
###### Sequential regulatory activity predictions with deep convolutional neural networks.
--------------------------------------------------------------------------------
## Variant analysis

<a name="sad"/>

### basenji_sad.py

Compute SNP Activity Difference (SAD) scores for SNPs in a VCF file.

| Arguments | Type | Description |
| --- | --- | --- |
| params_file | Text table | Model configuration parameters. |
| model file | TensorFlow Saver file | Trained Basenji model. |
| vcf_file | VCF | Variant Call Format file describing bi-allelic variants. |

--------------------------------------------------------------------------------
<a name="sed"/>

### basenji_sed.py

Compute SNP expression difference (SED) scores for SNPs in a VCF file.

| Arguments | Type | Description |
| --- | --- | --- |
| params_file | Text table | Model configuration parameters. |
| model file | TensorFlow Saver file | Trained Basenji model. |
| genes_hdf5_file | HDF5 | HDF5 file with gene sequences and descriptions. |
| vcf_file | VCF | Variant Call Format file describing bi-allelic variants. |

--------------------------------------------------------------------------------
<a name="sat_vcf"/>

### basenji_sat_vcf.py

Perform an in silico saturated mutagenesis of the sequences surrounding SNPS in a VCF file.

| Arguments | Type | Description |
| --- | --- | --- |
| params_file | Text table | Model configuration parameters. |
| model file | TensorFlow Saver file | Trained Basenji model. |
| vcf_file | VCF | Variant Call Format file describing bi-allelic variants. |
