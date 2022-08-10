## Manuscript models and data

[Cross-species regulatory sequence activity prediction. PLoS Comp Bio 7/2020.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008050)

*get_models.sh* - Download the saved TensorFlow models for human and mouse targets

Download the training/validation/test TFRecords	(319 Gb) from https://console.cloud.google.com/storage/browser/basenji_barnyard/data. Note that due to very large costs, we had to switch this to requester pays. 

Scikit-learn random forest SNP classifiers for Mendelian disease and GWAS complex traits available from https://console.cloud.google.com/storage/browser/basenji_barnyard/sad/classifiers/.
Restore models using joblib.load.

All 1000 Genomes variant scores for human and mouse available from https://console.cloud.google.com/storage/browser/basenji_barnyard/sad/human/ and https://console.cloud.google.com/storage/browser/basenji_barnyard/sad/mouse/.

Autism variant scores available from https://console.cloud.google.com/storage/browser/basenji_barnyard/sad/autism/.
