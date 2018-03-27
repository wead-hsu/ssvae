import theano
import numpy as np
from lasagne.layers import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__all__ = [
    "WordDropoutLayer",
    "ShiftLayer"
]

class WordDropoutLayer(Layer):
    def __init__(self, incoming, keep_rate = 1.0,**kwargs):
        super(WordDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        if keep_rate <= 0.0:
            self.keep_rate = 0.0
        elif keep_rate >= 1.0:
            self.keep_rate = 1.0
        else:
            self.keep_rate = keep_rate
        self.along_axis = along_axis
    def get_output_for(self, input, deterministic=False, **kwargs):
        '''
        when train, deter = False, keep_rate = 1.0 return input
        when test deter = True, 1)keep_rate = 0.0 return zeros, 2)else return input
        '''
        if (deterministic and self.keep_rate != 0.0) or self.keep_rate == 1.0:
            return input

        keep_shape = input_shape[:-1]
        keep_position = srng.binomial(keep_shape, n=1, p= self.keep_rate, dtype=theano.config.floatX)
        return input * keep_position[:,:,None]

class ShiftLayer(Layer):
    def __init__(self, incoming, shift_dir = 1, along_axis = 1, **kwargs):
        '''
        if shift_dir > 0 then the head is filled with zeros
        else the tail is filled with zeros
        '''
        super(ShiftLayer, self).__init__(incoming, **kwargs)
        self.shift_dir = shift_dir
        self.along_axis = along_axis

    def get_output_for(self, input, deterministic=False, **kwargs):
        '''
        this is only used as an function in vae.
        '''
        x_shift = T.zeros_like( input )
        x_shift = T.set_subtensor(x_shift[:,1:,:] , x[:,:-1,:])
        return x_shift










