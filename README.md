<img src="docs/basset_image.png" width="200">

# Basenji
#### Sequential regulatory activity predictions with deep convolutional neural networks.

Basenji provides researchers with tools to:
1. Train deep convolutional neural networks to predict regulatory activity along potentially very long DNA sequences
2. Score variants according to their predicted influence on regulatory activity across the sequence and/or for specific genes.
3. Annotate the distal regulatory elements that influence gene activity.
4. Annotate the specific nucleotides that drive regulatory element function.

---------------------------------------------------------------------------------------------------
### Basset successor

I'll contain to maintain the predecessor to this package--[Basset](https://github.com/davek44/Basset). However, this package offers numerous improvements and generalizations to that method. I'll be using this codebase for all of my ongoing work.

1. Basenji is built on TensorFlow, while Basset was written in Torch. TensorFlow offers myriad benefits, including distributed computing and a strong developer community.
2. Basenji makes predictions in bins across the sequences you provide. You could replicate Basset by simply providing smaller sequences and binning the target for the entire sequence.
3. Basenji intends to predict quantitative signal using regression loss functions, rather than binary signal using classification loss functions. But the classification mode is still present.

---------------------------------------------------------------------------------------------------
### Installation

Basenji has a variety of scientific computing dependencies. I highly recommend the [Anaconda distribution](https://www.continuum.io/downloads). The only library missing is pysam, which you can install through Anaconda or manually from [here](https://code.google.com/p/pysam/). If you don't want to use Anaconda, check out the full list of dependencies [here](docs/requirements.md).

Once you have the dependencies, run
```
    python setup.py develop
```

To verify the install, launch python and run
```
    import basenji
```

---------------------------------------------------------------------------------------------------
### Documentation

Basenji is under active development, so don't hesitate to ask for clarifications or additional features, documentation, or tutorials.

- [File specifications](docs/file_specs.md)
  - [BED](docs/file_specs.md#bed)
  - [Table](docs/file_specs.md#table)
  - [HDF5](docs/file_specs.md#hdf5)
  - [Model](docs/file_specs.md#model)
- [Preprocess](docs/preprocess.md)
  - [preprocess_features.py](docs/preprocess.md#preprocess_features.py)
  - [seq_hdf5.py](docs/preprocess.md#seq_hdf.py)
  - [basset_sample.py](docs/preprocess.md#basset_sample.py)
- [Learning](docs/learning.md)
  - [basset_train.lua](docs/learning.md#train)
  - [basset_test.lua](docs/learning.md#test)
  - [basset_predict.lua](docs/learning.md#predict)
- [Visualization](docs/visualization.md)
  - [basset_motifs.py](docs/visualization.md#motifs)
  - [basset_motifs_infl.py](docs/visualization.md#infl)
  - [basset_sat.py](docs/visualization.md#sat)
  - [basset_sat_vcf.py](docs/visualization.md#sat_vcf)
  - [basset_sad.py](docs/visualization.md#sad)

---------------------------------------------------------------------------------------------------
### Tutorials

These are a work in progress, so forgive incompleteness for the moment. If there's a task that you're interested in that I haven't included, feel free to post it as an Issue at the top.

- Preprocess
  - [Prepare the ENCODE and Epigenomics Roadmap compendium from scratch.](tutorials/prepare_compendium.ipynb)
  - [Prepare new dataset(s) by adding to a compendium.](tutorials/new_data_many.ipynb)
  - [Prepare new dataset(s) in isolation.](tutorials/new_data_iso.ipynb)
- Train
  - [Train a model.](tutorials/train.md)
- Test
  - [Test a trained model.](tutorials/test.ipynb)
- Visualization
  - [Study the motifs learned by the model.](tutorials/motifs.ipynb)
  - [Execute an in silico saturated mutagenesis](tutorials/sat_mut.ipynb)
  - [Compute SNP Accessibility Difference profiles.](tutorials/sad.ipynb)