You can find the official Enformer implementation here: https://github.com/google-deepmind/deepmind-research/tree/master/enformer

However, I have now brought the components into this repository, too. In this directory, I added a parameters json file that roughly matches the Enformer, with a few minor tweaks that I like. Namely, the positional encodings use central mask only, attention pooling is replaced with max pooling, the optimization settings are intended for smaller batch sizes, L2 weight decay is applied throughout, and the filter counts don't match exactly.
