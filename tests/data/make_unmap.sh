#!/bin/sh

black=$HG19/mappability/wgEncodeDacMapabilityConsensusExcludable.bed
unmap=$HG19/mappability/unmap_t8_l64.bed

cat $black $unmap_micro | awk 'BEGIN {OFS="\t"} {print $1, $2, $3}' > unmap_cat.bed
bedtools sort -i unmap_cat.bed > unmap_sort.bed
bedtools merge -i unmap_sort.bed > unmap.bed

rm unmap_cat.bed
rm unmap_sort.bed
