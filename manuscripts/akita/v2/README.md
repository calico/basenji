In 2021, we trained an updated version of the Akita model on both human and mouse data simultaneously. After an additional hyperparameter search, we boosted the accuracy significantly. Rather than a single train/valid/test split, we divided the genome into eight folds and trained one model with each fold held out for testing (and another as validation).

The data and models are stored in Google Cloud Storage, so the easiest way to acquire it is via gsutil.

E.g.
`gsutil cp -r gs://basenji_hic/3-2021/data .`
`gsutil cp -r gs://basenji_hic/3-2021/models .`

The data directory has subdirectories for human and mouse, specifying the sequence boundaries, their division into folds, and tfrecords.

The models directory has subdirectories for each training run with the index number specifying the fold held out as test. The index plus one modulo eight was held out as validation. The train subdirectory contains the saved models, with the suffix 0 specifying human and suffix 1 specifying mouse. You can also find test set metrics.

Please reach out if these are useful to you, and/or you have questions about how to use them!