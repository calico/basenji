# Copyright 2017 Calico LLC
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

import sys

import pysam

################################################################################
# bed.py
#
# Methods to work with BED files.
################################################################################

def make_bed_seqs(bed_file, fasta_file, seq_len, stranded=False):
  """Return BED regions as sequences and regions as a list of coordinate
  tuples, extended to a specified length."""
  """Extract and extend BED sequences to seq_len."""
  fasta_open = pysam.Fastafile(fasta_file)

  seqs_dna = []
  seqs_coords = []

  for line in open(bed_file):
    a = line.split()
    chrm = a[0]
    start = int(float(a[1]))
    end = int(float(a[2]))
    if len(a) >= 6:
      strand = a[5]
    else:
      strand = '+'

    # determine sequence limits
    mid = (start + end) // 2
    seq_start = mid - seq_len//2
    seq_end = seq_start + seq_len

    # save
    if stranded:
      seqs_coords.append((chrm,seq_start,seq_end,strand))
    else:
      seqs_coords.append((chrm,seq_start,seq_end))

    # initialize sequence
    seq_dna = ''

    # add N's for left over reach
    if seq_start < 0:
      print('Adding %d Ns to %s:%d-%s' % \
          (-seq_start,chrm,start,end), file=sys.stderr)
      seq_dna = 'N'*(-seq_start)
      seq_start = 0

    # get dna
    seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()

    # add N's for right over reach
    if len(seq_dna) < seq_len:
      print('Adding %d Ns to %s:%d-%s' % \
          (seq_len-len(seq_dna),chrm,start,end), file=sys.stderr)
      seq_dna += 'N'*(seq_len-len(seq_dna))

    # reverse complement
    if stranded and strand == '-':
      seq_dna = dna_io.dna_rc(seq_dna)

    # append
    seqs_dna.append(seq_dna)

  fasta_open.close()

  return seqs_dna, seqs_coords


def read_bed_coords(bed_file, seq_len):
  """Return BED regions as a list of coordinate
  tuples, extended to a specified length."""
  seqs_coords = []

  for line in open(bed_file):
    a = line.split()
    chrm = a[0]
    start = int(float(a[1]))
    end = int(float(a[2]))

    # determine sequence limits
    mid = (start + end) // 2
    seq_start = mid - seq_len//2
    seq_end = seq_start + seq_len

    # save
    seqs_coords.append((chrm,seq_start,seq_end))

  return seqs_coords
