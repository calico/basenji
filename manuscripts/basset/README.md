## Basset-style peak prediction in Basenji

[Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks. Genome Research 5/2016.](https://genome.cshlp.org/content/26/7/990.short)

*get_dnase.sh* - Download DNase peak files from ENCODE and Roadmap.
*make_dataset.sh* - Recommended parameters to generate a Basset-style single peak annotation dataset.

*params_basset.json* - Recommended parameters to train a Basset-style model. Note, these aren't the exact parameters used in that paper; they represent a roughly benchmarked improvement.

Note that I implemented the flatten/dense penultimate layers with a "valid" padded convolution. You have to specify the sequence length, pooling, and width of this convolution carefully to match your preprocessing of the peaks above. I recommend this approach because the flatten/dense layer can contain a huge majority of the model's parameters if designed thoughtlessly. I wouldn't go beyond a kernel width of 5-10; instead, pool your representation further. If you're having trouble getting this to work, reach out, and I'll lend a hand.

*basenji_train.py -k -o train_basset --rc params_basset.json data_basset*
