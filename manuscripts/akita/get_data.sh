#!/bin/sh

wget https://storage.googleapis.com/basenji_hic/1m/data/contigs.bed
wget https://storage.googleapis.com/basenji_hic/1m/data/sequences.bed

mkdir -p tfrecords
cd tfrecords
for i in $(seq 0 27)
do
    echo $i
    wget "https://storage.googleapis.com/basenji_hic/1m/data/tfrecords/train-$i.tfr"
done
for i in 0 1
do
    wget "https://storage.googleapis.com/basenji_hic/1m/data/tfrecords/valid-$i.tfr"
done
for i in 0 1
do
    wget "https://storage.googleapis.com/basenji_hic/1m/data/tfrecords/test-$i.tfr"
done
