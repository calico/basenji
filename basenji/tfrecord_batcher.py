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

import dna_io
import tensorflow as tf


def tf_record_dataset(tfr_data_file, batch_size, seq_depth, num_targets,
                      target_width, shuffle, trim_eos):
  """Load TFRecord format data.

     Args:
       tfr_data_file: TFRecord format file
       batch_size: batch_size
       seq_depth: vocabulary size of the inputs (4 for raw DNA)
       num_targets: number of targets at each target sequence location
       target_width: length of the target sequence
       shuffle: whether the batcher should shuffle the data
       trim_eos: whether to trim the final token from the inputs
     Returns:
       A dict with the following tensors:
         sequence: [batch_size, sequence_length, seq_depth]
         label: [batch_size, num_targets, target_width]
         na: [batch_size, num_targets]
  """
  inputs_name = 'inputs'
  targets_name = 'targets'

  dataset = tf.contrib.data.TFRecordDataset([tfr_data_file])
  features = {
      inputs_name: tf.VarLenFeature(tf.int64),
      targets_name: tf.VarLenFeature(tf.float32)
  }

  def _parse(example_proto):

    parsed_features = tf.parse_single_example(example_proto, features=features)

    seq = tf.cast(parsed_features[inputs_name].values, tf.int32)
    if(trim_eos):
      # useful because tensor2tensor preprocessing pads with an EOS
      seq = seq[:-1]

    seq = tf.one_hot(seq, seq_depth)

    label = tf.cast(parsed_features[targets_name].values, tf.float32)

    seq_n = tf.cast(tf.equal(tf.reduce_sum(seq, axis=1), 0), tf.float32)
    seq = tf.cast(seq, tf.float32)

    seq_n /= float(seq_depth)
    seq_n = tf.tile(tf.expand_dims(seq_n, axis=1), [1, seq_depth])
    seq += seq_n

    label = tf.reshape(label, [target_width, num_targets])
    na = tf.zeros(label.shape[:-1], dtype=tf.bool)

    return {'sequence': seq, 'label': label, 'na': na}

  dataset = dataset.map(_parse)
  dataset = dataset.repeat()
  if shuffle:
    dataset.shuffle(buffer_size=150)

  dataset = dataset.batch(batch_size)
  return dataset


class TFRecordBatcher(object):
  """Load TFRecord format data. Many args are unused and for API-compatibility.

     Args:
       tfr_data_file: TFRecord format file
       load_targets: whether to load targets (unused)
       seq_length: length of the input sequences (unused)
       seq_depth: vocabulary size of the inputs (4 for raw DNA)
       target_width: length of the target sequence
       num_targets: number of targets at each target sequence location
       NAf: (unused)
       batch_size: batch_size
       pool_width: width of pooling layers (unused)
       shuffle: whether the batcher should shuffle the data
       trim_eos: whether to trim the final token from the inputs
  """

  def __init__(self,
               tfr_data_file,
               load_targets,
               seq_length,
               seq_depth,
               target_width,
               num_targets,
               NAf=None,
               batch_size=64,
               pool_width=1,
               shuffle=False,
               trim_eos=True):

    self.session = None

    dataset = tf_record_dataset(tfr_data_file, batch_size, seq_depth,
                                num_targets, target_width, shuffle, trim_eos)

    self.iterator = dataset.make_initializable_iterator()
    self._next_element = self.iterator.get_next()

  def initialize(self, sess):
    sess.run(self.iterator.initializer)

  def next(self, rc=False, shift=0):
    try:
      d = self.session.run(self._next_element)

      Xb = d['sequence']
      Yb = d['label']
      NAb = d['na']
      Nb = Xb.shape[0]

      # reverse complement
      if rc:
        if Xb is not None:
          Xb = dna_io.hot1_augment(Xb, rc, shift)
        if Yb is not None:
          Yb = Yb[:, ::-1, :]
        if NAb is not None:
          NAb = NAb[:, ::-1]

      return Xb, Yb, NAb, Nb

    except tf.errors.OutOfRangeError:
      return None, None, None, None

  def reset(self):
    return self.initialize(self.session)
