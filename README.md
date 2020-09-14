<img src="docs/basset_image.png" width="200">

# Basenji
#### Sequential regulatory activity predictions with deep convolutional neural networks.

Basenji provides researchers with tools to:
1. Train deep convolutional neural networks to predict regulatory activity along very long chromosome-scale DNA sequences
2. Score variants according to their predicted influence on regulatory activity across the sequence and/or for specific genes.
3. Annotate the distal regulatory elements that influence gene activity.
4. Annotate the specific nucleotides that drive regulatory element function.

## Akita
#### 3D genome folding predictions with deep convolutional neural networks.

Akita provides researchers with tools to:
1. Train deep convolutional neural networks to predict 2D contact maps along very long chromosome-scale DNA sequences
2. Score variants according to their predicted influence on contact maps across the sequence and/or for specific genes.
3. Annotate the specific nucleotides that drive genome folding.

---------------------------------------------------------------------------------------------------
### Basset successor

This codebase offers numerous improvements and generalizations to its predecessor [Basset](https://github.com/davek44/Basset), and I'll be using it for all of my ongoing work. Here are the salient changes.

1. Basenji makes predictions in bins across the sequences you provide. You could replicate Basset's peak classification by simply providing smaller sequences and binning the target for the entire sequence.
2. Basenji intends to predict quantitative signal using regression loss functions, rather than binary signal using classification loss functions.
3. Basenji is built on [TensorFlow](https://www.tensorflow.org/), which offers myriad benefits, including distributed computing and a large and adaptive developer community.

However, this codebase is general enough to implement the Basset model, too. I have instructions for how to do that [here](manuscripts/basset).


---------------------------------------------------------------------------------------------------
### Installation

Basenji/Akita were developed with Python3 and a variety of scientific computing dependencies, which you can see and install via requirements.txt for pip and environment.yml for [Anaconda](https://www.continuum.io/downloads). For each case, we kept TensorFlow separate to allow you to choose the install method that works best for you. The codebase is compatible with the latest TensorFlow 2, but should also work with 1.15.

Run the following to install dependencies and Basenji with Anaconda.
```
    conda env create -f environment.yml
    conda install tensorflow (or tensorflow-gpu)
    python setup.py develop --no-deps
```

Alternatively, if you want to guarantee working versions of each dependency, you can install via a fully pre-specified environment.
```
    conda env create -f prespecified.yml
    conda install tensorflow (or tensorflow-gpu)
    python setup.py develop --no-deps
```

Or the following to install dependencies and Basenji with pip and setuptools.
```
    python setup.py develop
    pip install tensorflow (or tensorflow-gpu)
```

Then we recommend setting the following environmental variables.
```
  export BASENJIDIR=~/code/Basenji
  export PATH=$BASENJIDIR/bin:$PATH
  export PYTHONPATH=$BASENJIDIR/bin:$PYTHONPATH
```

To verify the install, launch python and run
```
    import basenji
```


---------------------------------------------------------------------------------------------------
### Manuscripts

Models and (links to) data studied in various manuscripts are available in the [manuscripts](manuscripts) directory.


---------------------------------------------------------------------------------------------------
### Documentation

At this stage, Basenji is something in between personal research code and accessible software for wide use. The primary challenge is uncertainty in what the best role for this type of toolkit is going to be in functional genomics and statistical genetics. The computational requirements don't make it easy either. Thus, this package is under active development, and I encourage anyone to get in touch to relate your experience and request clarifications or additional features, documentation, or tutorials.

- [Preprocess](docs/preprocess.md)
  - [bam_cov.py](docs/preprocess.md#bam_cov)
  - [basenji_hdf5_single.py](docs/preprocess.md#hdf5_single)
  - [basenji_hdf5_cluster.py](docs/preprocess.md#hdf5_cluster)
  - [basenji_hdf5_genes.py](docs/preprocess.md#hdf5_genes)
- [Train](docs/train.md)
  - [basenji_train.py](docs/train.md#train)
- [Accuracy](docs/accuracy.md)
  - [basenji_test.py](docs/accuracy.md#test)
  - [basenji_test_genes.py](docs/accuracy.md#test_genes)
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
- Train/test
  - [Train and test a model.](tutorials/train_test.ipynb)
- Gene expression
  - [Test gene expression predictions.](tutorials/genes.ipynb)
- Study
  - [Execute an in silico saturated mutagenesis](tutorials/sat_mut.ipynb)
  - [Compute SNP Activity Difference (SAD) and Expression Difference (SED) scores.](tutorials/sad.ipynb)
