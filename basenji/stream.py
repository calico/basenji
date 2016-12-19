#!/usr/bin/env python

class PredStream:
    ''' Interface to acquire predictions via a buffered stream mechanism
         rather than getting them all at once and using excessive memory. '''

    def __init__(self, sess, model, seqs_1hot, stream_length):
        self.sess = sess
        self.model = model

        self.seqs_1hot = seqs_1hot

        self.stream_length = stream_length
        self.stream_start = 0
        self.stream_end = 0

        if self.stream_length % self.model.batch_size != 0:
            print('Make the stream length a multiple of the batch size', file=sys.stderr)
            exit(1)


    def __getitem__(self, i):
        # acquire predictions, if needed
        if i >= self.stream_end:
            self.stream_start = self.stream_end
            self.stream_end = min(self.stream_start + self.stream_length, self.seqs_1hot.shape[0])

            # subset sequences
            stream_seqs_1hot = self.seqs_1hot[self.stream_start:self.stream_end]

            # initialize batcher
            batcher = basenji.batcher.Batcher(stream_seqs_1hot, batch_size=self.model.batch_size)

            # predict
            self.stream_preds = self.model.predict(self.sess, batcher, rc_avg=False)

        return self.stream_preds[i - self.stream_start]