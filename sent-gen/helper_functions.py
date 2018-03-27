# helper functions
import os
import lasagne
import theano
from theano import tensor as T
from theano import ifelse
import numpy as np

########################################################################################################################
#Optmizer...
# copied from https://raw.githubusercontent.com/bartvm/blocks/master/blocks/algorithms/__init__.py
########################################################################################################################
def shared_floatx(value, name=None, borrow=False, dtype=None):
    #Transform a value into a shared variable of type floatX.
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
            name=name,
            borrow=borrow)

def l2_norm(tensors):
    #Computes the total L2 norm of a set of tensors.
    flattened = [T.as_tensor_variable(t).flatten() for t in tensors]
    flattened = [(t if t.ndim > 0 else t.dimshuffle('x'))
                 for t in flattened]
    joined = T.join(0, *flattened)
    return T.sqrt(T.sqr(joined).sum())

def step_clipping(steps, threshold, to_zero=False):
    #Rescales an entire step if its L2 norm exceeds a threshold.
    threshold = shared_floatx(threshold)
    norm = l2_norm(steps)   # return total norm
    if to_zero:
        print("clipping to zero")
        scale = 1e-8  # smallstep
    else:
        scale = threshold / norm
    multiplier = T.switch(norm < threshold, 1.0, scale)

    return [step*multiplier for step in steps], norm, multiplier


def adam(all_grads, all_params,learning_rate=0.0002, beta1=0.1, beta2=0.001, epsilon=1e-8, decay_factor=1-1e-8, max_lr = 0.01):
    #Adam optimizer as described in iederik Kingma, Jimmy Ba, *Adam: A Method for Stochastic Optimization*,http://arxiv.org/abs/1412.6980
    name_prefix = 'Adam_'
    time = shared_floatx(0., 'time')
    t1 = time + 1
    updates, steps = [], []
    for param, g in zip(all_params, all_grads):
        if param.name:
            tmp_name = name_prefix + param.name + '_'
        else:
            tmp_name = name_prefix
        mean = shared_floatx(param.get_value()*0., tmp_name + 'mean')
        variance = shared_floatx(param.get_value()*0., tmp_name + 'variance')

        learning_rate = (learning_rate * T.sqrt((1. - (1. - beta2)**t1)) / (1. - (1. - beta1)**t1))
        learning_rate = T.switch( learning_rate < max_lr, learning_rate,max_lr )
        beta_1t = 1 - (1 - beta1) * decay_factor ** (t1 - 1)
        mean_t = beta_1t * g + (1. - beta_1t) * mean
        variance_t = (beta2 * T.sqr(g) + (1. - beta2) * variance)
        step = (learning_rate * mean_t / (T.sqrt(variance_t) + epsilon))

        updates.append((mean, mean_t))
        updates.append((variance, variance_t))
        updates.append((param, param - step))
        steps.append(step)

    updates.append((time, t1))
    return updates, steps

def adadelta(all_grads, all_params, decay_rate = 0.95, epsilon = 1e-6):
    steps, updates = [], []
    name_prefix = 'AdaDelta_'
    for param, g in zip(all_params, all_grads):
        if param.name:
            tmp_name = name_prefix + param.name + '_'
        else:
            tmp_name = name_prefix
        mean_square_step_tm1 = shared_floatx(param.get_value() * 0., tmp_name + 'mean_square_step')
        mean_square_delta_x_tm1 = shared_floatx(param.get_value() * 0., tmp_name + 'mean_square_delta_x')

        mean_square_step_t = (decay_rate * mean_square_step_tm1 + (1 - decay_rate) * T.sqr(g))
        rms_delta_x_tm1 = T.sqrt(mean_square_delta_x_tm1 + epsilon)
        rms_step_t = T.sqrt(mean_square_step_t + epsilon)
        delta_x_t = rms_delta_x_tm1 / rms_step_t * g
        mean_square_delta_x_t = (decay_rate * mean_square_delta_x_tm1 + (1 - decay_rate) * T.sqr(delta_x_t))

        steps.append(delta_x_t)
        updates.append((mean_square_step_tm1, mean_square_step_t))
        updates.append((mean_square_delta_x_tm1, mean_square_delta_x_t))
        updates.append((param, param - delta_x_t))

    return updates, steps

def adagrad(all_grads, all_params, learning_rate = 0.002, epsilon = 1e-6):
    steps, updates = [], []
    name_prefix = 'AdaGrad_'
    for param, g in zip(all_params, all_grads):
        if param.name:
            tmp_name = name_prefix + param.name + '_'
        else:
            tmp_name = name_prefix
        ssq = shared_floatx(param.get_value() *0., tmp_name + 'sqs')

        ssq_t = (T.sqr(g) + ssq)
        step = (learning_rate * g / (T.sqrt(ssq_t) + epsilon))

        steps.append(step)
        updates.append((ssq, ssq_t))
        updates.append((param, param - step))

    return updates, steps


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
        data_batches = [self.process_func(data_n[batch]) for data_n in self.data]
        return data_batches


def threaded_generator(generator, num_cached=50):
    # this code is writte by jan Schluter
    # copied from https://github.com/benanne/Lasagne/issues/12
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()

########################################################################################################################
#Other functions
########################################################################################################################

def save_params(layer, save_path):
    """
    param : [p.get_value(borrow=True) for p in all_params] or use lasagne.layer .get_params()
    example: save_params( l_vae, "param/1")
    """
    all_params = lasagne.layers.get_all_params(layer)
    param = [p.get_value(borrow=True) for p in all_params]
    print "Saving parameters to" + save_path
    import cPickle
    cPickle.dump(param, open(save_path, "w"))

def load_params(layer, load_path):
    """
    layer: lasagne layer with .get_params()
    """
    import cPickle
    print "Loading parameters from " + load_path
    values =cPickle.load(open(load_path,"r"))
    params = lasagne.layers.get_all_params(layer)
    if len(params) != len(values):
        raise ValueError("mismatch: got %d values to set %d parameters" %
                         (len(values), len(params)))

    for p, v in zip(params, values):
        if p.get_value().shape != v.shape:
            raise ValueError("shape mismatch")
        else:
            p.set_value(v)

def log_write_line(log_path , strline, mode_str):
    with open(log_path, mode_str) as f:
        f.write(strline)
        f.write('\n')
    return