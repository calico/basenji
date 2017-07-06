<img src="docs/basset_image.png" width="200">

# Basenji
#### Sequential regulatory activity predictions with deep convolutional neural networks.

Basenji provides researchers with tools to:
1. Train deep convolutional neural networks to predict regulatory activity along very long chromosome-scale DNA sequences
2. Score variants according to their predicted influence on regulatory activity across the sequence and/or for specific genes.
3. Annotate the distal regulatory elements that influence gene activity.
4. Annotate the specific nucleotides that drive regulatory element function.

---------------------------------------------------------------------------------------------------
### Basset successor

This codebase offers numerous improvements and generalizations to its predecessor [Basset](https://github.com/davek44/Basset), and I'll be using it for all of my ongoing work. Here are the salient changes.

1. Basenji makes predictions in bins across the sequences you provide. You could replicate Basset's peak classification by simply providing smaller sequences and binning the target for the entire sequence.
2. Basenji intends to predict quantitative signal using regression loss functions, rather than binary signal using classification loss functions.
3. Basenji is built on [TensorFlow](https://www.tensorflow.org/), which offers myriad benefits, including distributed computing and a large and adaptive developer community.


---------------------------------------------------------------------------------------------------
### Installation

Basenji has a variety of scientific computing dependencies, which you can see within the setup.py file. I highly recommend the [Anaconda python distribution](https://www.continuum.io/downloads), which contains most of them.

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

At this stage, Basenji is something in between personal research code and an accessible software for wide use. The primary challenge is uncertainty what the best role for this type of toolkit is going to be in functional genomics and statistical genetics. Thus, Basenji is under active development, and I encourage anyone to get in touch to relate your experience and request clarifications or additional features, documentation, or tutorials.

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