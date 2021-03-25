## Basset-style peak prediction in Basenji

[Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks. Genome Research 5/2016.](https://genome.cshlp.org/content/26/7/990.short)

*get_dnase.sh* - Download DNase peak files from ENCODE and Roadmap.

*make_dataset.sh* - Recommended parameters to generate a Basset-style single peak annotation dataset.

*params_basset.json* - Recommended parameters to train a Basset-style model. Note, these aren't the exact parameters used in that paper; they represent a roughly benchmarked improvement.

*basenji_train.py -k -o train_basset params_basset.json data_basset*
