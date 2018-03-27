import sys
import  codecs
import os
import cPickle as pkl
import gzip
import numpy as np

class BatchGenerator():
    def __init__(self, data, num_batches, extra_process = None):
        self.data, self.data_size, self.extra_process = data, num_batches, extra_process
        # self.data is A LIST of numpy array[(N, None,...),(N,..)...], with same size
        self.data_size = self.data[0].shape[0]
        self.batch_size = int(self.data_size / self.num_batches)
        self.cnt = 0
        self.random_index = np.random.permutation(self.data_num)

    def next_batch(self):
        self.cnt = self.cnt % self.num_batches
        if self.cnt == 0:
            self.random_index = np.random.shuffle(self.random_index)
        if self.cnt == self.num_batches -1:
            slice_idx = self.random_index[self.cnt*self.batch_size:]
        else:
            slice_idx = self.random_index[self.cnt*self.batch_size:(self.cnt+1)*self.batch_size]
        self.cnt += 1
        slice = [d[slice_idx] for d in self.data]
        if self.extra_process is not None:
            return self.extra_process(slice)
        else:
            return slice

