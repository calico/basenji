#!/usr/bin/env python

################################################################################
# gtf_kv
#
# Convert the last gtf section of key/value pairs into a dict.
################################################################################
def gtf_kv(s):
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


################################################################################
# read_genes
#
# Parse a gtf file and return a set of Gene objects in a hash keyed by the
# id given.
#
# Note: assumes exons only.
################################################################################
def read_genes(gtf_file, key_id='transcript_id', sort=True):
    genes = {}

    gtf_in = open(gtf_file)

    # ignore header
    line = gtf_in.readline()
    while line[:2] == '##':
        line = gtf_in.readline()

    while line:
        a = line.split('\t')
        if a[2] in ['exon','CDS']:
            kv = gtf_kv(a[8])
            if not kv[key_id] in genes:
                genes[kv[key_id]] = Gene(a[0], a[6], kv)

            if a[2] == 'exon':
                genes[kv[key_id]].add_exon(int(a[3]), int(a[4]), sort=sort)
            elif a[2] == 'CDS':
                genes[kv[key_id]].add_cds(int(a[3]), int(a[4]), sort=sort)

        line = gtf_in.readline()

    gtf_in.close()

    return genes


################################################################################
# Gene
################################################################################
class Gene:
    def __init__(self, chrom, strand, kv):
        self.chrom = chrom
        self.strand = strand
        self.kv = kv
        self.exons = []
        self.cds = []

    def add_cds(self, start, end, sort=True):
        self.cds.append(Exon(start,end))
        if sort and len(self.cds) > 1 and self.cds[-2].end > start:
            #print >> sys.stderr, 'CDS are not sorted - %s %d %d' % (self.chrom,self.start,self.end)
            self.cds.sort()

    def add_exon(self, start, end, sort=True):
        self.exons.append(Exon(start,end))
        if sort and len(self.exons) > 1 and self.exons[-2].end > start:
            #print >> sys.stderr, 'Warning: exons are not sorted - %s' % kv_gtf(self.kv)
            self.exons.sort()

    def tss(self):
        if self.strand == '-':
            return self.exons[-1].end
        else:
            return self.exons[0].start

    def __str__(self):
        return '%s %s %s %s' % (self.chrom, self.strand, kv_gtf(self.kv), ','.join([ex.__str__() for ex in self.exons]))


################################################################################
# Exon
################################################################################
class Exon:
    def __init__(self, start, end):
        self.start = start
        self.end = end

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
        return 'exon(%d-%d)' % (self.start,self.end)
