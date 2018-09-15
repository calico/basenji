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
import gzip
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pysam

import basenji.dna_io
"""vcf.py

Methods and classes to support .vcf SNP analysis.
"""


def cap_allele(allele, cap=5):
  """ Cap the length of an allele in the figures """
  if len(allele) > cap:
    allele = allele[:cap] + '*'
  return allele


def intersect_seqs_snps(vcf_file, gene_seqs, vision_p=1):
  """ Intersect a VCF file with a list of sequence coordinates.

    In
     vcf_file:
     gene_seqs: list of GeneSeq's
     vision_p: proportion of sequences visible to center genes.

    Out
     seqs_snps: list of list mapping segment indexes to overlapping SNP indexes
    """

  # print segments to BED
  # hash segments to indexes
  seq_temp = tempfile.NamedTemporaryFile()
  seq_bed_file = seq_temp.name
  seq_bed_out = open(seq_bed_file, 'w')
  seq_indexes = {}
  for si in range(len(gene_seqs)):
    gs = gene_seqs[si]
    gene_seq_key = (gs.chr, gs.start)
    seq_indexes[gene_seq_key] = si
    print('%s\t%d\t%d' % (gs.chr, gs.start, gs.end), file=seq_bed_out)
  seq_bed_out.close()

  # hash SNPs to indexes
  snp_indexes = {}
  si = 0

  vcf_in = open(vcf_file)
  line = vcf_in.readline()
  while line[0] == '#':
    line = vcf_in.readline()
  while line:
    a = line.split()
    snp_id = a[2]
    if snp_id in snp_indexes:
      raise Exception('Duplicate SNP id %s will break the script' % snp_id)
    snp_indexes[snp_id] = si
    si += 1
    line = vcf_in.readline()
  vcf_in.close()

  # initialize list of lists
  seqs_snps = []
  for _ in range(len(gene_seqs)):
    seqs_snps.append([])

  # intersect
  p = subprocess.Popen(
      'bedtools intersect -wo -a %s -b %s' % (vcf_file, seq_bed_file),
      shell=True,
      stdout=subprocess.PIPE)
  for line in p.stdout:
    line = line.decode('UTF-8')
    a = line.split()
    pos = int(a[1])
    snp_id = a[2]
    seq_chrom = a[-4]
    seq_start = int(a[-3])
    seq_end = int(a[-2])
    seq_key = (seq_chrom, seq_start)

    vision_buffer = (seq_end - seq_start) * (1 - vision_p) // 2
    if seq_start + vision_buffer < pos < seq_end - vision_buffer:
      seqs_snps[seq_indexes[seq_key]].append(snp_indexes[snp_id])

  p.communicate()

  return seqs_snps


def intersect_snps_seqs(vcf_file, seq_coords, vision_p=1):
  """ Intersect a VCF file with a list of sequence coordinates.

    In
     vcf_file:
     seq_coords: list of sequence coordinates
     vision_p: proportion of sequences visible to center genes.

    Out
     snp_segs: list of list mapping SNP indexes to overlapping sequence indexes
    """
  # print segments to BED
  # hash segments to indexes
  seg_temp = tempfile.NamedTemporaryFile()
  seg_bed_file = seg_temp.name
  seg_bed_out = open(seg_bed_file, 'w')
  segment_indexes = {}

  for si in range(len(seq_coords)):
    segment_indexes[seq_coords[si]] = si
    print('%s\t%d\t%d' % seq_coords[si], file=seg_bed_out)

  seg_bed_out.close()

  # hash SNPs to indexes
  snp_indexes = {}
  si = 0

  vcf_in = open(vcf_file)
  line = vcf_in.readline()
  while line[0] == '#':
    line = vcf_in.readline()
  while line:
    a = line.split()
    snp_id = a[2]
    if snp_id in snp_indexes:
      raise Exception('Duplicate SNP id %s will break the script' % snp_id)
    snp_indexes[snp_id] = si
    si += 1
    line = vcf_in.readline()
  vcf_in.close()

  # initialize list of lists
  snp_segs = []
  for i in range(len(snp_indexes)):
    snp_segs.append([])

  # intersect
  p = subprocess.Popen(
      'bedtools intersect -wo -a %s -b %s' % (vcf_file, seg_bed_file),
      shell=True,
      stdout=subprocess.PIPE)
  for line in p.stdout:
    line = line.decode('UTF-8')
    a = line.split()
    pos = int(a[1])
    snp_id = a[2]
    seg_chrom = a[-4]
    seg_start = int(a[-3])
    seg_end = int(a[-2])
    seg_key = (seg_chrom, seg_start, seg_end)

    vision_buffer = (seg_end - seg_start) * (1 - vision_p) // 2
    if seg_start + vision_buffer < pos < seg_end - vision_buffer:
      snp_segs[snp_indexes[snp_id]].append(segment_indexes[seg_key])

  p.communicate()

  return snp_segs


def snp_seq1(snp, seq_len, genome_open):
  """ Produce a one hot coded sequences for a SNP.

    Attrs:
        snp [SNP] :
        seq_len (int) : sequence length to code
        genome_open (File) : open genome FASTA file

    Return:
        seq_vecs_list [array] : list of one hot coded sequences surrounding the
        SNP
    """
  left_len = seq_len // 2 - 1
  right_len = seq_len // 2

  # initialize one hot coded vector list
  seq_vecs_list = []

  # specify positions in GFF-style 1-based
  seq_start = snp.pos - left_len
  seq_end = snp.pos + right_len + max(0,
                                      len(snp.ref_allele) - snp.longest_alt())

  # extract sequence as BED style
  if seq_start < 0:
    seq = 'N'*(1-seq_start) + genome_open.fetch(snp.chr, 0, seq_end).upper()
  else:
    seq = genome_open.fetch(snp.chr, seq_start - 1, seq_end).upper()

  # extend to full length
  if len(seq) < seq_end - seq_start:
    seq += 'N' * (seq_end - seq_start - len(seq))

  # verify that ref allele matches ref sequence
  seq_ref = seq[left_len:left_len + len(snp.ref_allele)]
  ref_found = True
  if seq_ref != snp.ref_allele:

    # search for reference allele in alternatives
    ref_found = False

    # for each alternative allele
    for alt_al in snp.alt_alleles:

      # grab reference sequence matching alt length
      seq_ref_alt = seq[left_len:left_len + len(alt_al)]
      if seq_ref_alt == alt_al:
        # found it!
        ref_found = True

        # warn user
        print(
            'WARNING: %s - alt (as opposed to ref) allele matches reference genome; changing reference genome to match.'
            % (snp.rsid),
            file=sys.stderr)

        # remove alt allele and include ref allele
        seq = seq[:left_len] + snp.ref_allele + seq[left_len + len(alt_al):]
        break

  if not ref_found:
    print('WARNING: %s - reference genome does not match any allele' % snp.rsid, file=sys.stderr)

  else:
    # one hot code ref allele
    seq_vecs_ref, seq_ref = dna_length_1hot(seq, seq_len)
    seq_vecs_list.append(seq_vecs_ref)

    for alt_al in snp.alt_alleles:
      # remove ref allele and include alt allele
      seq_alt = seq[:left_len] + alt_al + seq[left_len + len(snp.ref_allele):]

      # one hot code
      seq_vecs_alt, seq_alt = dna_length_1hot(seq_alt, seq_len)
      seq_vecs_list.append(seq_vecs_alt)

  return seq_vecs_list


def snps_seq1(snps, seq_len, genome_fasta, return_seqs=False):
  """ Produce an array of one hot coded sequences for a list of SNPs.

    Attrs:
        snps [SNP] : list of SNPs
        seq_len (int) : sequence length to code
        genome_fasta (str) : genome FASTA file

    Return:
        seq_vecs (array) : one hot coded sequences surrounding the SNPs
        seq_headers [str] : headers for sequences
        seq_snps [SNP] : list of used SNPs
    """
  left_len = seq_len // 2 - 1
  right_len = seq_len // 2

  # initialize one hot coded vector list
  seq_vecs_list = []

  # save successful SNPs
  seq_snps = []

  # save sequence strings, too
  seqs = []

  # name sequences
  seq_headers = []

  # open genome FASTA
  genome_open = pysam.Fastafile(genome_fasta)

  for snp in snps:
    # specify positions in GFF-style 1-based
    seq_start = snp.pos - left_len
    seq_end = snp.pos + right_len + max(0,
                                        len(snp.ref_allele) - snp.longest_alt())

    # extract sequence as BED style
    if seq_start < 0:
      seq = 'N' * (-seq_start) + genome_open.fetch(snp.chr, 0,
                                                   seq_end).upper()
    else:
      seq = genome_open.fetch(snp.chr, seq_start - 1, seq_end).upper()

    # extend to full length
    if len(seq) < seq_end - seq_start:
      seq += 'N' * (seq_end - seq_start - len(seq))

    # verify that ref allele matches ref sequence
    seq_ref = seq[left_len:left_len + len(snp.ref_allele)]
    if seq_ref != snp.ref_allele:

      # search for reference allele in alternatives
      ref_found = False

      # for each alternative allele
      for alt_al in snp.alt_alleles:

        # grab reference sequence matching alt length
        seq_ref_alt = seq[left_len:left_len + len(alt_al)]
        if seq_ref_alt == alt_al:
          # found it!
          ref_found = True

          # warn user
          print(
              'WARNING: %s - alt (as opposed to ref) allele matches reference genome; changing reference genome to match.'
              % (snp.rsid),
              file=sys.stderr)

          # remove alt allele and include ref allele
          seq = seq[:left_len] + snp.ref_allele + seq[left_len + len(alt_al):]
          break

      if not ref_found:
        print(
            'WARNING: %s - reference genome does not match any allele; skipping'
            % (snp.rsid),
            file=sys.stderr)
        continue

    seq_snps.append(snp)

    # one hot code ref allele
    seq_vecs_ref, seq_ref = dna_length_1hot(seq, seq_len)
    seq_vecs_list.append(seq_vecs_ref)
    if return_seqs:
      seqs.append(seq_ref)

    # name ref allele
    seq_headers.append('%s_%s' % (snp.rsid, cap_allele(snp.ref_allele)))

    for alt_al in snp.alt_alleles:
      # remove ref allele and include alt allele
      seq_alt = seq[:left_len] + alt_al + seq[left_len + len(snp.ref_allele):]

      # one hot code
      seq_vecs_alt, seq_alt = dna_length_1hot(seq_alt, seq_len)
      seq_vecs_list.append(seq_vecs_alt)
      if return_seqs:
        seqs.append(seq_alt)  # not using right now

      # name
      seq_headers.append('%s_%s' % (snp.rsid, cap_allele(alt_al)))

  # convert to array
  seq_vecs = np.array(seq_vecs_list)

  if return_seqs:
    return seq_vecs, seq_headers, seq_snps, seqs
  else:
    return seq_vecs, seq_headers, seq_snps


def snps2_seq1(snps, seq_len, genome1_fasta, genome2_fasta, return_seqs=False):
  """ Produce an array of one hot coded sequences for a list of SNPs.

    Attrs:
        snps [SNP] : list of SNPs
        seq_len (int) : sequence length to code
        genome_fasta (str) : major allele genome FASTA file
        genome2_fasta (str) : minor allele genome FASTA file

    Return:
        seq_vecs (array) : one hot coded sequences surrounding the SNPs
        seq_headers [str] : headers for sequences
        seq_snps [SNP] : list of used SNPs
    """
  left_len = seq_len // 2 - 1
  right_len = seq_len // 2

  # open genome FASTA
  genome1 = pysam.Fastafile(genome1_fasta)
  genome2 = pysam.Fastafile(genome2_fasta)

  # initialize one hot coded vector list
  seq_vecs_list = []

  # save successful SNPs
  seq_snps = []

  # save sequence strings, too
  seqs = []

  # name sequences
  seq_headers = []

  for snp in snps:
    if len(snp.alt_alleles) > 1:
      raise Exception(
          'Major/minor genome mode requires only two alleles: %s' % snp.rsid)

    alt_al = snp.alt_alleles[0]

    # specify positions in GFF-style 1-based
    seq_start = snp.pos - left_len
    seq_end = snp.pos + right_len + len(snp.ref_allele)

    # extract sequence as BED style
    if seq_start < 0:
      seq_ref = 'N' * (-seq_start) + genome1.fetch(snp.chr, 0,
                                                   seq_end).upper()
    else:
      seq_ref = genome1.fetch(snp.chr, seq_start - 1, seq_end).upper()

    # extend to full length
    if len(seq_ref) < seq_end - seq_start:
      seq_ref += 'N' * (seq_end - seq_start - len(seq_ref))

    # verify that ref allele matches ref sequence
    seq_ref_snp = seq_ref[left_len:left_len + len(snp.ref_allele)]
    if seq_ref_snp != snp.ref_allele:
      raise Exception(
          'WARNING: Major allele SNP %s doesnt match reference genome: %s vs %s'
          % (snp.rsid, snp.ref_allele, seq_ref_snp))

    # specify positions in GFF-style 1-based
    seq_start = snp.pos2 - left_len
    seq_end = snp.pos2 + right_len + len(alt_al)

    # extract sequence as BED style
    if seq_start < 0:
      seq_alt = 'N' * (-seq_start) + genome2.fetch(snp.chr, 0,
                                                   seq_end).upper()
    else:
      seq_alt = genome2.fetch(snp.chr, seq_start - 1, seq_end).upper()

    # extend to full length
    if len(seq_alt) < seq_end - seq_start:
      seq_alt += 'N' * (seq_end - seq_start - len(seq_alt))

    # verify that ref allele matches ref sequence
    seq_alt_snp = seq_alt[left_len:left_len + len(alt_al)]
    if seq_alt_snp != alt_al:
      raise Exception(
          'WARNING: Minor allele SNP %s doesnt match reference genome: %s vs %s'
          % (snp.rsid, snp.alt_alleles[0], seq_alt_snp))

    seq_snps.append(snp)

    # one hot code ref allele
    seq_vecs_ref, seq_ref = dna_length_1hot(seq_ref, seq_len)
    seq_vecs_list.append(seq_vecs_ref)
    if return_seqs:
      seqs.append(seq_ref)

    # name ref allele
    seq_headers.append('%s_%s' % (snp.rsid, cap_allele(snp.ref_allele)))

    # one hot code alt allele
    seq_vecs_alt, seq_alt = dna_length_1hot(seq_alt, seq_len)
    seq_vecs_list.append(seq_vecs_alt)
    if return_seqs:
      seqs.append(seq_alt)

    # name
    seq_headers.append('%s_%s' % (snp.rsid, cap_allele(alt_al)))

  # convert to array
  seq_vecs = np.array(seq_vecs_list)

  if return_seqs:
    return seq_vecs, seq_headers, seq_snps, seqs
  else:
    return seq_vecs, seq_headers, seq_snps


def dna_length_1hot(seq, length):
  """ Adjust the sequence length and compute
        a 1hot coding. """

  if length < len(seq):
    # trim the sequence
    seq_trim = (len(seq) - length) // 2
    seq = seq[seq_trim:seq_trim + length]

  elif length > len(seq):
    # extend with N's
    nfront = (length - len(seq)) // 2
    nback = length - len(seq) - nfront
    seq = 'N' * nfront + seq + 'N' * nback

  seq_1hot = basenji.dna_io.dna_1hot(seq)

  return seq_1hot, seq


def vcf_snps(vcf_file, require_sorted=False, validate_ref_fasta=None, flip_ref=False, pos2=False):
  """ Load SNPs from a VCF file """
  if vcf_file[-3:] == '.gz':
    vcf_in = gzip.open(vcf_file, 'rt')
  else:
    vcf_in = open(vcf_file)

  # read through header
  line = vcf_in.readline()
  while line[0] == '#':
    line = vcf_in.readline()

  # to check sorted
  if require_sorted:
    seen_chrs = set()
    prev_chr = None
    prev_pos = -1

  # to check reference
  if validate_ref_fasta is not None:
    genome_open = pysam.Fastafile(validate_ref_fasta)

  # read in SNPs
  snps = []
  while line:
    snps.append(SNP(line, pos2))

    if require_sorted:
      if prev_chr is not None:
        # same chromosome
        if prev_chr == snps[-1].chr:
          if snps[-1].pos < prev_pos:
            print('Sorted VCF required. Mis-ordered position: %s' % line.rstrip(),
                  file=sys.stderr)
            exit(1)
        elif snps[-1].chr in seen_chrs:
          print('Sorted VCF required. Mis-ordered chromosome: %s' % line.rstrip(),
                 file=sys.stderr)
          exit(1)

      seen_chrs.add(snps[-1].chr)
      prev_chr = snps[-1].chr
      prev_pos = snps[-1].pos

    if validate_ref_fasta is not None:
      ref_n = len(snps[-1].ref_allele)
      snp_pos = snps[-1].pos-1
      ref_snp = genome_open.fetch(snps[-1].chr, snp_pos, snp_pos+ref_n)
      if snps[-1].ref_allele != ref_snp:
        if not flip_ref:
          # bail
          print('ERROR: %s does not match reference %s' % (snps[-1], ref_snp), file=sys.stderr)
          exit(1)

        else:
          alt_n = len(snps[-1].alt_alleles[0])
          ref_snp = genome_open.fetch(snps[-1].chr, snp_pos, snp_pos+alt_n)

          # if alt matches fasta reference
          if snps[-1].alt_alleles[0] == ref_snp:
            # flip alleles
            snps[-1].flip_alleles()

          else:
            # bail
            print('ERROR: %s does not match reference %s' % (snps[-1], ref_snp), file=sys.stderr)
            exit(1)

    line = vcf_in.readline()

  vcf_in.close()

  return snps


def vcf_sort(vcf_file):
  # move
  os.rename(vcf_file, '%s.tmp' % vcf_file)

  # print header
  vcf_out = open(vcf_file, 'w')
  print('##fileformat=VCFv4.0', file=vcf_out)
  vcf_out.close()

  # sort
  subprocess.call(
      'bedtools sort -i %s.tmp >> %s' % (vcf_file, vcf_file), shell=True)

  # clean
  os.remove('%s.tmp' % vcf_file)


class SNP:
  """ SNP

    Represent SNPs read in from a VCF file

    Attributes:
        vcf_line (str)
    """

  def __init__(self, vcf_line, pos2=False):
    a = vcf_line.split()
    if a[0].startswith('chr'):
      self.chr = a[0]
    else:
      self.chr = 'chr%s' % a[0]
    self.pos = int(a[1])
    self.rsid = a[2]
    self.ref_allele = a[3]
    self.alt_alleles = a[4].split(',')
    # self.alt_allele = self.alt_alleles[0]

    if self.rsid == '.':
      self.rsid = '%s:%d' % (self.chr, self.pos)

    self.pos2 = None
    if pos2:
      self.pos2 = int(a[5])

  def flip_alleles(self):
    """ Flip reference and first alt allele."""
    assert(len(self.alt_alleles) == 1)
    self.ref_allele, self.alt_alleles[0] = self.alt_alleles[0], self.ref_allele

  def get_alleles(self):
    """ Return a list of all alleles """
    alleles = [self.ref_allele] + self.alt_alleles
    return alleles

  def longest_alt(self):
    """ Return the longest alt allele. """
    return max([len(al) for al in self.alt_alleles])

  def __str__(self):
    return 'SNP(%s, %s:%d, %s/%s)' % (self.rsid, self.chr, self.pos,
                                      self.ref_allele,
                                      ','.join(self.alt_alleles))
