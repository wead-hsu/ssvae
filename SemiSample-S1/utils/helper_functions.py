# helper functions
import os
import lasagne
import theano
from theano import tensor as T
from theano import ifelse
import numpy as np
########################################################################################################################
#Data process
########################################################################################################################

def make_mask(x, l_seq = 200):
    '''
    this function calculate x and mask. if l_seq is given, then extract sentences shorter than l_seq
    '''
    new_x = []
    for i in x:
        r = 0
        s = i[r]
        while r+1 < len(i) and len(s) + len(i[r+1]) < l_seq:
            s = s + i[r+1]
            r += 1
        new_x.append(s)
    x = new_x

    x_len = []
    max_len = 0
    for s in x:
        x_len.append(len(s))
        if max_len < len(s):
            max_len = len(s)

    # append <EOS> to data
    xx = np.zeros([len(x), max_len + 1], dtype='int32')
    m = np.zeros([len(x), max_len + 1], dtype=theano.config.floatX)
    for i, s in enumerate(x):
        xx[i, :x_len[i]] = x[i]
        m[i, :x_len[i] + 1] = 1.0

    max_sent_length = min(l_seq, max_len)
    xx = xx[:, :max_sent_length]
    m = m[:, :max_sent_length]
    return xx, m

########################################################################################################################
#Batch Inerator
########################################################################################################################
class BatchIterator(object):
    """
     Cyclic Iterators over batch indexes. Permutes and restarts at end
    """

    def __init__(self, batch_indices, batchsize, data, testing=False, process_func=None):
        if isinstance(batch_indices, int):
            self.n = batch_indices
            self.batchidx = np.arange(batch_indices)
        else:
            self.n = len(batch_indices)
            self.batchidx = np.array(batch_indices)

        self.batchsize = batchsize
        self.testing = testing
        if process_func is None:
            process_func = lambda x:x
        self.process_func = process_func

        if not isinstance(data, (list, tuple)):
            data = [data]

        self.data = data
        if not self.testing:
            self.createindices = lambda: np.random.permutation(self.n)
        else: # testing == true
            assert self.n % self.batchsize == 0, "for testing n must be multiple of batch size"
            self.createindices = lambda: range(self.n)

        self.perm = self.createindices()
        assert self.n > self.batchsize

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def _get_permuted_batches(self,n_batches):
        # return a list of permuted batch indeces
        batches = []
        for i in range(n_batches):

            # extend random permuation if shorter than batchsize
            if len(self.perm) <= self.batchsize:
                new_perm = self.createindices()
                self.perm = np.hstack([self.perm, new_perm])

            batches.append(self.perm[:self.batchsize])
            self.perm = self.perm[self.batchsize:]
        return batches

    def next(self):
        batch = self._get_permuted_batches(1)[0]   # extract a single batch
        data_batches = [self.process_func([data_n[i] for i in batch]) for data_n in self.data]
        return data_batches

########################################################################################################################
#Other functions
########################################################################################################################

def log_write_line(log_path , strline, mode_str):
    with open(log_path, mode_str) as f:
        f.write(strline)
        f.write('\n')
    return