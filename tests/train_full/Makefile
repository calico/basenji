FASTA_HUMAN=$$HG38/assembly/ucsc/hg38.ml.fa
GAPS_HUMAN=$$HG38/assembly/ucsc/hg38_gaps.bed
UMAP_HUMAN=$$HG38//mappability/umap_k24_t10_l32.bed
BLACK_HUMAN=$$HG38/blacklist/hg38.blacklist.rep.bed

# stride_train 87339 = floor(262144/128/3) + 43
# stride_test 229376 = 262144 - 2*16384
OPTS=-p 32 -r 64 -w 128 --crop 16384 -l 262144 -t 0.12 -v 0.12 --soft --umap_t 0.5 --break 8388608 --stride_train 87339 --stride_test 229376

all: data/tfrecords/train-0-0.tfr

targets.txt: make_targets.py
	./make_targets.py

umap_human.bed:
	cat $(UMAP_HUMAN) $(BLACK_HUMAN) | awk 'BEGIN {OFS="\t"} {print $$1, $$2, $$3}' | bedtools sort -i - | bedtools merge -i - > umap_human.bed

data/tfrecords/train-0-0.tfr: targets.txt umap_human.bed
	basenji_data.py -b $(BLACK_HUMAN) -g $(GAPS_HUMAN) -u umap_human.bed $(OPTS) -o data $(FASTA_HUMAN) targets.txt
