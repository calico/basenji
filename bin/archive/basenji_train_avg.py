#!/usr/bin/env python
from optparse import OptionParser
import os

import slurm

'''
Name

Description...
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <seed_model> <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-e', dest='num_epochs',
      default=4, type='int',
      help='Number of epochs to train models [Default: %default]')
    parser.add_option('-n', dest='num_models',
      default=3, type='int',
      help='Number of models to train [Default: %default]')
    parser.add_option('-o', dest='out_dir',
      default='seqnn_avg',
      help='Output directory in which to train [Default: %default]')
    parser.add_option('-s', dest='num_steps',
      default=None, type='int',
      help='Number of steps to train models [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
      parser.error('Must provide parameters, seed model, and data')
    else:
      params_file = args[0]
      seed_model = args[1]
      data_file = args[2]

    if not os.path.isdir(options.out_dir):
      os.mkdir(options.out_dir)

    jobs = []

    for mi in range(options.num_models):
      model_dir = '%s/m%d' % (options.out_dir,mi)

      cmd = 'source activate py3_gpu;'
      cmd += ' basenji_train.py'
      cmd += ' --rc --shifts "3,2,1,0,-1,-2,-3"'
      cmd += ' --logdir %s' % model_dir
      cmd += ' --check_all'
      cmd += ' --num_train_epochs %d' % options.num_epochs
      cmd += ' --restart %s' % seed_model
      cmd += ' --params %s' % params_file
      cmd += ' --data %s' % data_file

      j = slurm.Job(cmd, name=model_dir,
        out_file='%s.out'%model_dir,
        err_file='%s.err'%model_dir,
        queue='gtx1080ti', gpu=1, cpu=1,
        time='4-0:0:0', mem=30000)

      jobs.append(j)

    slurm.multi_run(jobs, verbose=True)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
