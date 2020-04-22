## Akita: Predicting 3D genome folding from DNA sequence
--------------------------------------------------------------------------------

### Manuscript model and data

[Predicting 3D genome folding from DNA sequence. bioRxiv 10/2019.](https://www.biorxiv.org/content/10.1101/800060v1)

*get_model.sh* - Download the saved TensorFlow model.

*get_data.sh* - Download the training/validation/test TFRecords	(~10 Gb).

--------------------------------------------------------------------------------

### Explore predictions 

*browsable predictions for chr17*: https://resgen.io/gfudenberg/Akita-preds/views

*explore_model.ipynb*: Load and visualize predictions for the trained model. [link to nbviewer](https://nbviewer.jupyter.org/github/gfudenberg/basenji/blob/tf2_hic/manuscripts/akita/explore_model.ipynb)

Requires:
- *params.json* - Model configuration parameters.
- *model_best.h5* -  Trained TensorFlow model weights.
- *targets.txt* -  List of coolers the model was trained on.
- *sequences.bed* - List of sequences used for model training.
- *tfrecords* - Folder with tfrecords.

In addition to Basenji and its dependencies, this notebook requires [cooltools](https://github.com/mirnylab/cooltools). We recommend creating a fresh conda environment with numpy, scipy, pandas, jupyter, installing [Basenji](https://github.com/calico/basenji/tree/master/#installation), then installing cooltools:
```
pip install cooltools
```

--------------------------------------------------------------------------------

### Train new models

In addition to Basenji dependencies, generating training data for Akita requires:
- [astropy](https://docs.astropy.org/en/stable/install.html)
- [cooler](https://github.com/mirnylab/cooler/)
- [cooltools](https://github.com/mirnylab/cooltools)
- [intervaltree](https://pypi.org/project/intervaltree/)

We recommend installing these dependencies via pip as well.

*tutorial.ipynb* - prepare training dat and train a model. [link to nbviewer](https://nbviewer.jupyter.org/github/gfudenberg/basenji/blob/tf2_hic/manuscripts/akita/tutorial.ipynb)

Preparing training data and training models both follow the basenji syntax, albeit using *akita_data.py* and *akita_train.py*.

