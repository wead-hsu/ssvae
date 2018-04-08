# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init

class MeanMaskLayer(layers.MergeLayer):
    def __init__(self, incoming, mask_input, **kwargs):
        incomings = [incoming, mask_input]
        super(MeanMaskLayer, self).__init__(incomings)

    def get_output_for(self, inputs, **kwargs):
        '''
        sent_ebd, mask = inputs, with shape (batch_size, max_sent_length, ebd_dims)
                                            (batch_size, max_sent_length)
        return the mean pooling vector of each sentence with mask.
                                            (batch_size, ebd_dims)
        '''
        sent_ebd, mask_input =  inputs
        sent_mean = (sent_ebd * mask_input[:, :, None]).sum(axis = 1)
        sent_mean = sent_mean / mask_input.sum(axis=1)[:, None]
        return sent_mean

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2])



# this class is for getting the hidden output of sentence classification
# . Or used for sentence encoder
# there is a doubt whether mean operation is better for encoder
class MeanLstmLayer(layers.MergeLayer):
    def __init__(self, incoming, num_units, mask_input,
                 grad_clipping = 0, **kwargs):

        incomings = [incoming, mask_input]
        super(MeanLstmLayer, self).__init__(incomings)
        self.num_units = num_units
        self.lstm_layer = layers.LSTMLayer(incoming,
            num_units = self.num_units,
            mask_input = mask_input,
            grad_clipping = grad_clipping,
            **kwargs)

    def get_output_for(self, inputs, **kwargs):
        sent_input, mask_input =  inputs
        lstm_output = self.lstm_layer.get_output_for(inputs)
        lstm_mean = (lstm_output * mask_input[:, :, None]).sum(axis = 1)
        lstm_mean = lstm_mean / mask_input.sum(axis=1)[:, None]
        return lstm_mean


    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], self.num_units)


    def get_params(self,**kwargs):
        return self.lstm_layer.get_params(**kwargs)