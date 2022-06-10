# Copyright 2022 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from intervaltree import IntervalTree
import pybedtools

class GenomicInterval:
  def __init__(self, start, end, chrom=None, strand=None):
    self.start = start
    self.end = end
    self.chrom = chrom
    self.strand = strand

  def __eq__(self, other):
    return self.start == other.start

  def __lt__(self, other):
    return self.start < other.start

  def __cmp__(self, x):
    if self.start < x.start:
      return -1
    elif self.start > x.start:
      return 1
    else:
      return 0

  def __str__(self):
    if self.chrom is None:
      label = '[%d-%d]' % (self.start, self.end)
    else:
      label =  '%s:%d-%d' % (self.chrom, self.start, self.end)
    return label


class Gene:
  """Class for managing genes in an isoform-agnostic way, taking
  the union of exons across isoforms."""
  def __init__(self, chrom, strand, kv):
    self.chrom = chrom
    self.strand = strand
    self.kv = kv
    self.exons = IntervalTree()

  def add_exon(self, start, end):
    """BED 0-indexing assumed."""
    self.exons[start:end] = True

  def get_exons(self):
    self.exons.merge_overlaps()
    return sorted(self.exons)

  def span(self):
    exon_starts = [exon.begin for exon in self.exons]
    exon_ends = [exon.end for exon in self.exons]
    return min(exon_starts), max(exon_ends)


class Transcriptome:
  def __init__(self, gtf_file):
    self.genes = {}
    self.read_gtf(gtf_file)

  def read_gtf(self, gtf_file):
    if gtf_file[-3:] == '.gz':
      gtf_in = gzip.open(gtf_file, 'rt')
    else:   
      gtf_in = open(gtf_file)

    # ignore header
    line = gtf_in.readline()
    while line[0] == '#':
        line = gtf_in.readline()

    while line:
      a = line.split('\t')
      if a[2] == 'exon':
        chrom = a[0]
        start = int(a[3])
        end = int(a[4])
        strand = a[6]
        kv = gtf_kv(a[8])
        gene_id = kv['gene_id']

        # initialize gene
        if gene_id not in self.genes:
          self.genes[gene_id] = Gene(chrom, strand, kv)

        # add exon
        self.genes[gene_id].add_exon(start-1, end)

      line = gtf_in.readline()

    gtf_in.close()

  def bedtool_exon(self):
    # assemble sequence bedtool
    bed_lines = []
    for gene_id, gene in self.genes.items():
      for exon in gene.get_exons():
        exon_line = '%s %d %d %s . %s' % (gene.chrom, exon.begin, exon.end, gene_id, gene.strand)
        bed_lines.append(exon_line)
    genes_bedt = pybedtools.BedTool('\n'.join(bed_lines), from_string=True)
    return genes_bedt

  def bedtool_span(self):
    # assemble sequence bedtool
    bed_lines = []
    for gene_id, gene in self.genes.items():
      gene_start, gene_end = gene.span()
      span_line = '%s %d %d %s . %s' % (gene.chrom, gene_start, gene_end, gene_id, gene.strand)
      bed_lines.append(span_line)
    genes_bedt = pybedtools.BedTool('\n'.join(bed_lines), from_string=True)
    return genes_bedt

  def write_bed_exon(self, bed_file):
    pass

  def write_bed_span(self, bed_file):
    pass


################################################################################
# Methods
################################################################################
def gtf_kv(s):
  """Convert the last gtf section of key/value pairs into a dict."""
  d = {}

  a = s.split(';')
  for key_val in a:
    if key_val.strip():
      eq_i = key_val.find('=')
      if eq_i != -1 and key_val[eq_i-1] != '"':
        kvs = key_val.split('=')
      else:
        kvs = key_val.split()

      key = kvs[0]
      if kvs[1][0] == '"' and kvs[-1][-1] == '"':
        val = (' '.join(kvs[1:]))[1:-1].strip()
      else:
        val = (' '.join(kvs[1:])).strip()

      d[key] = val

  return d