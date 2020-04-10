# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function

import sys
import pysam

################################################################################
# genome.py
#
# Methods to interact with genome information.
################################################################################


def load_chromosomes(genome_file):
  """ Load genome segments from either a FASTA file or
          chromosome length table. """

  # is genome_file FASTA or (chrom,start,end) table?
  file_fasta = (open(genome_file).readline()[0] == '>')

  chrom_segments = {}

  if file_fasta:
    fasta_open = pysam.Fastafile(genome_file)
    for i in range(len(fasta_open.references)):
      chrom_segments[fasta_open.references[i]] = [(0, fasta_open.lengths[i])]
    fasta_open.close()

  else:
    for line in open(genome_file):
      a = line.split()
      chrom_segments[a[0]] = [(0, int(a[1]))]

  return chrom_segments


def split_contigs(chrom_segments, gaps_file):
  """ Split the assembly up into contigs defined by the gaps.

    Args:
      chrom_segments: dict mapping chromosome names to lists of (start,end)
      gaps_file: file specifying assembly gaps

    Returns:
      chrom_segments: same, with segments broken by the assembly gaps.
    """

  chrom_events = {}

  # add known segments
  for chrom in chrom_segments:
    if len(chrom_segments[chrom]) > 1:
      print(
          "I've made a terrible mistake...regarding the length of chrom_segments[%s]"
          % chrom,
          file=sys.stderr)
      exit(1)
    cstart, cend = chrom_segments[chrom][0]
    chrom_events.setdefault(chrom, []).append((cstart, 'Cstart'))
    chrom_events[chrom].append((cend, 'cend'))

  # add gaps
  for line in open(gaps_file):
    a = line.split()
    chrom = a[0]
    gstart = int(a[1])
    gend = int(a[2])

    # consider only if its in our genome
    if chrom in chrom_events:
      chrom_events[chrom].append((gstart, 'gstart'))
      chrom_events[chrom].append((gend, 'Gend'))

  for chrom in chrom_events:
    # sort
    chrom_events[chrom].sort()

    # read out segments
    chrom_segments[chrom] = []
    for i in range(len(chrom_events[chrom]) - 1):
      pos1, event1 = chrom_events[chrom][i]
      pos2, event2 = chrom_events[chrom][i + 1]

      event1 = event1.lower()
      event2 = event2.lower()

      shipit = False
      if event1 == 'cstart' and event2 == 'cend':
        shipit = True
      elif event1 == 'cstart' and event2 == 'gstart':
        shipit = True
      elif event1 == 'gend' and event2 == 'gstart':
        shipit = True
      elif event1 == 'gend' and event2 == 'cend':
        shipit = True
      elif event1 == 'gstart' and event2 == 'gend':
        pass
      else:
        print(
            "I'm confused by this event ordering: %s - %s" % (event1, event2),
            file=sys.stderr)
        exit(1)

      if shipit and pos1 < pos2:
        chrom_segments[chrom].append((pos1, pos2))

  return chrom_segments
