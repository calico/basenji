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

At this stage, Basenji is something in between personal research code and accessible software for wide use. The primary challenge is uncertainty what the best role for this type of toolkit is going to be in functional genomics and statistical genetics. The computational requirements don't make it easy either. Thus, this package is under active development, and I encourage anyone to get in touch to relate your experience and request clarifications or additional features, documentation, or tutorials.

- [Preprocess](docs/preprocess.md)
  - [bam_cov.py](docs/preprocess.md#bam_cov.py)
  - [basenji_hdf5_single.py](docs/preprocess.md#basenji_single.py)
  - [basenji_hdf5_cluster.py](docs/preprocess.md#basenji_cluster.py)
  - [basenji_genes.py](docs/preprocess.md#basenji_genes.py)
- [Train](docs/train.md)
  - [basenji_train.py](docs/train.md#train)
- [Accuracy](docs/accuracy.md)
  - [basenji_test.py](docs/accuracy.md#test)
  - [basenji_test_genes.py](docs/accuracy.md#test_genes)
  - [basenji_predict.py](docs/accuracy.md#predict)
- [Regulatory element analysis](docs/regulatory.md)
  - [basenji_motifs.py](docs/regulatory.md#motifs)
  - [basenji_sat.py](docs/regulatory.md#sat)
  - [basenji_map.py](docs/regulatory.md#map)
- [Variant analysis](docs/variants.md)
  - [basenji_sad.py](docs/variants.md#sad)
  - [basenji_sed.py](docs/variants.md#sed)
  - [basenji_sat_vcf.py](docs/variants.md#sat_vcf)

---------------------------------------------------------------------------------------------------
### Tutorials

These are a work in progress, so forgive incompleteness for the moment. If there's a task that you're interested in that I haven't included, feel free to post it as an Issue at the top.

- Preprocess
  - [Preprocess new datasets for training.](tutorials/preprocess.ipynb)
- Train
  - [Train a model.](tutorials/train.md)
- Test
  - [Test a trained model.](tutorials/test.ipynb)
- Visualization
  - [Study the motifs learned by the model.](tutorials/motifs.ipynb)
  - [Execute an in silico saturated mutagenesis](tutorials/sat_mut.ipynb)
  - [Compute SNP Accessibility Difference profiles.](tutorials/sad.ipynb)