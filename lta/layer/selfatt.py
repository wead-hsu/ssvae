# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init

class SelfAttentionLayer(layers.MergeLayer):
    def __init__(self, incoming, mask_input, num_units, **kwargs):
        incomings = [incoming, mask_input]
        super(SelfAttentionLayer, self).__init__(incomings)
        self.num_units = num_units
        self.l_h = layers.DenseLayer(incoming, num_units, nonlinearities.tanh)
        self.l_a = layers.DenseLayer(self.l_h, 1, None)

    def get_output_for(self, inputs, **kwargs):
        x, m = inputs
        logits_att = self.l_h.get_output_for(x)
        logits_att = self.l_a.get_output_for(logits_att)

        max_logit_att = T.max(logits_att - 1e20*(1-m[:,:,None]), axis=1)[:,None,:]
        logits_att = T.exp(logits_att - max_logit_att) * m[:, :, None]
        weights = logits_att / T.sum(logits_att, axis=1)[:, None, :]
        h = T.sum(x * weights, axis=1)
        self.weights = weights[:, :, 0] #damn it, this breaks the formulation of lasagne
        return h

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], self.num_units)
