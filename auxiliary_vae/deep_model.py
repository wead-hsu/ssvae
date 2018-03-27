import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.objectives import *
from lasagne.layers import *
from lasagne import init
from lasagne.nonlinearities import *
from lasagne.updates import total_norm_constraint
from lasagne.updates import adam
from layer.sample import *
from layer.variationalcost import *
from layer.mylstm import *
from layer.sclstm import *
from layer.meanlstm import *
import copy

# coded by Haoze Sun 2016.2

class DeepModel():
    def __init__(self,
                #model params used in stage initialization
                enc_num_units, dec_num_units, dim_z, word_ebd_dims, word_dict_size,
                dim_y = 1, dim_a = 1, drop_out = 0.10, keep_rate = 0.8, bnalpha = 0.1,
                sample_unlabel = False,
                #train/evalue params that unchange during train
                grad_clipping = 1.0, max_norm = 20.0,
                cost_beta = 5.0, lr = 0.0008, beta1 = 0.9, beta2 = 0.999
                #other params that unchange
                ):
        #########  init params ####################
        print 'Using SDGM!!!!!!!!!!!!'
        self.enc_num_units, self.dec_num_units = enc_num_units, dec_num_units
        self.word_ebd_dims, self.word_dict_size= word_ebd_dims, word_dict_size
        self.dimy, self.dima, self.dimz = dim_y, dim_a, dim_z
        self.drop_out, self.keep_rate, self.bnalpha = drop_out, keep_rate, bnalpha
        self.sample_unlabel = sample_unlabel
        self.grad_clipping, self.max_norm = grad_clipping, max_norm

        self.cost_beta = cost_beta #0.1 ~ 2.0
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        ############################################

    def build_model(self, use_mean_lstm = True, old_version = True,
                    #init params used during setup stage
                    transf = lasagne.nonlinearities.tanh, # density layer active function
                    word_ebd_init = init.Normal(1e-6),
                    b_init = init.Normal(1e-4), W_init = init.GlorotNormal(),
                    W_init_act = init.GlorotNormal()):
        self.transf = transf
        ############## build model ############
        self.l_sents_in = InputLayer((None,None))
        self.l_mask_in = InputLayer((None,None))
        self.l_label_in = InputLayer((None,self.dimy)) #for unlabel data, y is generated from classifier,else y is a parameter in trainning
        self.l_z_in = InputLayer((None, self.dimz)) #samples in generation model
        self.l_dec_cell_in = InputLayer((None, self.dec_num_units)) #used in one step beam search
        self.l_dec_hid_in = InputLayer((None, self.dec_num_units)) #used in one step beam search
        self.l_dec_input_word_in = InputLayer((None, None, self.word_ebd_dims)) #batch_size * sent_length(max lstm steps) * word_ebd_dims
        self.l_dec_out_in = InputLayer((None,None, self.dec_num_units))
        ###word embedding layers
        self.l_ebd =  EmbeddingLayer(self.l_sents_in,self.word_dict_size, self.word_ebd_dims, W = word_ebd_init,
                                         name = 'EbdLayer' )
        self.l_ebd_drop = DropoutLayer(self.l_ebd, p = self.drop_out,
                                       name = 'EbdDropoutLayer') #no params; input: batch_size*sent_length*word_ebd_dims
        ####################encoder lstm layers######################################
        self.l_x = DropoutLayer( LSTMLayer(self.l_ebd_drop, num_units = self.enc_num_units, mask_input = self.l_mask_in,
                                                    grad_clipping = self.grad_clipping, only_return_final = True, name='EncLSTMLayer'),
                                                    p = self.drop_out, name='EncLSTMLayer') #LSTM for classifier mean pooling is better?
        if use_mean_lstm:
            print 'Using mean pooling for classifier!!!!!!!!!!!!!!!'
            self.l_c0 = MeanLstmLayer(self.l_ebd_drop, num_units = self.enc_num_units, mask_input = self.l_mask_in,
                                            grad_clipping = self.grad_clipping, name='ClassLSTMLayer')
            self.l_c = DropoutLayer(self.l_c0, p = self.drop_out, name='ClassLSTMLayer')

        else:
            self.l_c0 = LSTMLayer(self.l_ebd_drop, num_units = self.enc_num_units, mask_input = self.l_mask_in,
                                            grad_clipping = self.grad_clipping, only_return_final = True, name='ClassLSTMLayer')
            self.l_c = DropoutLayer(self.l_c0, p = self.drop_out, name='ClassLSTMLayer')
        #----------------- auxiliary q(a|x) ###########################################
        if old_version:
            self.l_x_to_a = DropoutLayer( batch_norm( DenseLayer(self.l_x, num_units= self.dima,
                                                W = W_init_act, b = b_init,
                                                nonlinearity= self.transf, name = 'x_to_a_old'),
                                                alpha = self.bnalpha, name = 'x_to_a_old'),p = self.drop_out, name = 'x_to_a_old')
        else:
            print 'Using new version of model!!!!!!!!!!'
            self.l_mean_pooling =DropoutLayer(MeanMaskLayer(self.l_ebd_drop, self.l_mask_in, name='mean_pooling'),
                                              p = self.drop_out, name = 'mean_pooling')
            self.l_x_to_a = DropoutLayer( batch_norm( DenseLayer(self.l_mean_pooling, num_units= self.dima,
                                                W = W_init_act, b = b_init,
                                                nonlinearity= self.transf, name = 'x_to_a_new'),
                                                alpha = self.bnalpha, name = 'x_to_a_new'),p = self.drop_out, name = 'x_to_a_new')

        self.l_a_mu = DenseLayer(self.l_x_to_a, self.dima, W = W_init, b = b_init, nonlinearity=None, name = 'a_mu') #Linear without active functions
        self.l_a_var = DenseLayer(self.l_x_to_a, self.dima, W = W_init,b = b_init, nonlinearity=None, name = 'a_var')
        self.l_a = SimpleSampleLayer(self.l_a_mu, self.l_a_var, name= 'a_sample') #no params
        ################# Classifier q(y|a,x) #####################################
        self.l_ax = ConcatLayer([self.l_c, self.l_a], axis=1, name = 'Concat_ax') #no params
        self.l_ax_to_y = DropoutLayer( batch_norm( DenseLayer(self.l_ax, num_units=30,
                                        W = W_init_act, b= b_init,
                                        nonlinearity= self.transf,name = 'ax_to_y'),
                                        alpha = self.bnalpha,name = 'ax_to_y'), p = self.drop_out, name = 'ax_to_y')
        self.l_y = DenseLayer(self.l_ax, num_units= self.dimy,W=W_init, b = b_init,
                              nonlinearity= softmax, name='y_classifier' )
        #################### sample q(z|a,x,y) ####################################
        self.l_axy = ConcatLayer([self.l_x, self.l_a, self.l_label_in], axis=1, name = 'Concat_axy') #no params first use l_label_in
        self.l_axy_to_z = DropoutLayer( batch_norm( DenseLayer(self.l_axy, num_units= self.dimz,
                                                    W = W_init_act, b= b_init,
                                                    nonlinearity= self.transf, name='axy_to_z'),
                                                    alpha = self.bnalpha, name='axy_to_z'), p = self.drop_out, name='axy_to_z')
        self.l_z_mu = DenseLayer(self.l_axy_to_z, self.dimz,W=W_init, b=b_init, nonlinearity=None, name='z_mu') #Linear without active functions
        self.l_z_var = DenseLayer(self.l_axy_to_z, self.dimz,W=W_init, b=b_init, nonlinearity=None, name='z_var') #Linear without active functions
        self.l_z = SimpleSampleLayer(self.l_z_mu, self.l_z_var, name ='z_sample')

        ################## generative model, we use 'u' to stand 'a' in paper #####
        self.l_yz = ConcatLayer([self.l_label_in, self.l_z_in], axis = 1, name='Concat_yz') #l_z_in layer is used in beam search
        self.l_yz_to_u = DropoutLayer( batch_norm( DenseLayer(self.l_yz, num_units= self.dima,
                                                    W = W_init_act, b= b_init,
                                                    nonlinearity= self.transf, name='yz_to_u' ),
                                                    alpha = self.bnalpha, name='yz_to_u'), p = self.drop_out,name='yz_to_u')
        self.l_u_mu = DenseLayer(self.l_yz_to_u, self.dima,W=W_init, b=b_init, nonlinearity=None, name='u_mu')
        self.l_u_var = DenseLayer(self.l_yz_to_u, self.dima,W=W_init, b=b_init, nonlinearity=None, name='u_var')
        self.l_u = SimpleSampleLayer(self.l_u_mu, self.l_u_var, name='u_sample')
        ######################## hiddens for LSTM #################
        self.l_uyz = ConcatLayer([self.l_yz, self.l_u], axis = 1, name='Concat_uyz')
        self.l_hid = batch_norm( DenseLayer(self.l_uyz, num_units= self.dec_num_units,
                                                    W = W_init_act, b= b_init,
                                                    nonlinearity= self.transf, name ='LmHidInit'),
                                                    alpha = self.bnalpha, name ='LmHidInit') #init of hidden has no dropout
        ######################## dec lm ###################
        self.l_lm =  ScLSTMLayer(incoming=self.l_dec_input_word_in, num_units= self.dec_num_units,da_init=self.l_label_in,
                                cell_init=self.l_dec_cell_in,  hid_init=self.l_dec_hid_in, mask_input= self.l_mask_in,
                                grad_clipping = self.grad_clipping, name='ScLSTMLayer') #cell, hid used in beam search, shape(batch_size,sent_length,dec_num_units)
        ######################## softmax results ###################
        self.l_recons_x = DenseLayer(DropoutLayer(ReshapeLayer(self.l_dec_out_in, shape=(-1, self.dec_num_units), name='ScLSTMLayer'),
                                     p = self.drop_out, name='ScLSTMLayer'),#output shape:( batch_size*sent_length,dec_num_units)
                                num_units = self.word_dict_size, W=W_init, b=b_init,nonlinearity = softmax, name='recons_x')#(batch_size*sent_length, word_dict_size)
        '''
        when we want to get recons_x in training or test:
                    dec_out = get_output(self.l_lm, {self.l_dec_cell_in:cell_init, self.l_dec_hid_in:hid,
                            self.l_dec_input_word_in:ebd_shift_worddrop,self.l_mask_in:mask},
                           deterministic=dev_stage) #(batch_size*sent_length, word_dict_size)
                    prob_x = get_output(self.l_recons_x, {self.l_dec_out_in:dec_out},
                           deterministic=dev_stage)
        when we want to get mid result of lstm layer:
                    the same
        '''

    def get_params(self, only_trainable = False):
        '''
        all the params are divided into two groups: 1) trainable 2) untrainable etc. batch normal layers
         when save the model, we need all the params, only_train = False
         when calculate the gradients, only trainable parameters are used

         auxiliary vae is combined with a classifier (l_y), encoder (l_z) and decoder (l_recons_x)
        '''
        ##################################################
        #Layer tree map, end with input layers
        #0. reconstructor:       l_recons_x <--- l_dec_out_in
        #1. decoder:             l_lm <--- l_dec_input_word, l_dec_cell, l_dec_hid, l_mask_in, l_label
        #2. samples to LSTM hid: l_hid, l_uyz, l_u,..., l_yz_to_u  <--- l_label_in, l_z_in
        #3. reparameter z:       l_z,...,l_a, l_x, l_ebd <--- l_label_in, l_sents_in, l_mask_in
        #4. classifier y:        l_y, l_ax_to_y, l_c  <--- l_sents_in, l_mask_in

        if only_trainable:
            return get_all_params([self.l_y,self.l_z,self.l_hid,self.l_lm,self.l_recons_x],
                                             trainable = True)
        else: # for batch normalization layers params are untrainable
            return get_all_params([self.l_y,self.l_z,self.l_hid,self.l_lm,self.l_recons_x])

    def save_model(self, save_path ):
        all_params = self.get_params()
        param = [p.get_value(borrow=True) for p in all_params]
        print "Saving parameters to" + save_path
        import cPickle
        cPickle.dump(param, open(save_path, "wb"))

    def load_model(self, load_path):
        import cPickle
        print "Loading parameters from " + load_path
        values =cPickle.load(open(load_path,"rb"))
        params = self.get_params()
        if len(params) != len(values):
            raise ValueError("mismatch: got %d values to set %d parameters" %
                             (len(values), len(params)))

        for p, v in zip(params, values):
            if p.get_value().shape != v.shape:
                raise ValueError("shape mismatch")
            else:
                p.set_value(v)

    def _inference(self, incomings, dev_stage=False):
        print 'Inference Model Q.'
        s, mask, y = incomings
        return get_output([self.l_ebd, self.l_y,
                           self.l_a_mu,self.l_a_var,self.l_a,
                           self.l_z_mu, self.l_z_var,self.l_z],
                          {self.l_sents_in:s, self.l_mask_in:mask, self.l_label_in:y},
                          deterministic=dev_stage)

    def _word_dropout(self, input, keep_rate, deterministic=False, **kwargs ):
        '''
        input: (batch_size, max_sent_length, word_ebd_dim)
        when train, deter = False, keep_rate = 1.0 return input
        when test deter = True, 1)keep_rate = 0.0 return zeros, 2)else return input
        '''
        print 'Word Drop Out Function.'
        if keep_rate <= 0.0:
            keep_rate = 0.0
        elif keep_rate >= 1.0:
            keep_rate = 1.0
        if (deterministic and keep_rate != 0.0) or keep_rate == 1.0:
            return input, T.sum(0)

        _srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        keep_position = _srng.binomial((input.shape[0],input.shape[1]), n=1, p= keep_rate, dtype=theano.config.floatX)
        return input * keep_position[:,:,None], T.sum(1.0-keep_position).astype('int64')

    def _shift(self, input, **kwargs):
        '''
        input: (batch_size, max_sent_length, word_ebd_dim)
        '''
        print 'Word Shift Function.'
        x_shift = T.zeros_like( input )
        x_shift = T.set_subtensor(x_shift[:,1:,:] , input[:,:-1,:])
        return x_shift

    def _gen_lm_init(self, incomings, dev_stage=False):
        '''
        generate u and hid initial state form [y,z]
        '''
        print 'From <y,z> to u and hid.'
        y, z = incomings
        return get_output([self.l_u_mu, self.l_u_var, self.l_u, self.l_hid],
                          {self.l_z_in:z, self.l_label_in:y},
                          deterministic=dev_stage)

    def _calc_kl(self, incomings, return_mode = 'by_batch'):
        '''
        return_mode: by_batch, (batch_size,)
        return_mode: by_unit   (unit_size,)
        '''
        print 'KL loss function'
        y, z_mu, z_var, a_mu, a_var, u_mu, u_var = incomings
        label_priori = T.ones(y.shape,dtype=theano.config.floatX) / y.shape[1].astype(theano.config.floatX)
        kl_p0y = categorical_crossentropy(label_priori,y) #vector batch_size
        kl_p0z, kl_p0z_by_unit = GaussianMarginalLogDensity([z_mu, z_var], normal_priori=True)
        kl_qz, kl_qz_by_unit = GaussianMarginalLogDensity([z_mu, z_var], normal_priori=False)
        kl_qa, kl_qa_by_unit = GaussianMarginalLogDensity([a_mu, a_var], normal_priori=False)
        kl_pa, kl_pa_by_unit = GaussianQLogPDensity([a_mu, a_var, u_mu, u_var])

        if return_mode == 'by_batch':
            return kl_p0y, kl_p0z, kl_qz, kl_qa, kl_pa
        else:
            return kl_p0z_by_unit, kl_qz_by_unit, kl_qa_by_unit, kl_pa_by_unit

    def cost_label(self, incomings, dev_stage = False, return_mode = 'sum'):
        '''
        basic vae cost for both label data
        incomings - [ s, mask , y]
        s -  (batch_size * sents_length(lstm steps) ) index of words theano sym
        mask - the same as s, theano sym
        y - (batch_size * ydim) one-hot for labeled data, y is an input data; for unlabel data, y is generated form
            classifier theano sym
        train_stage - bool var indicate the stage: train/dev.test
        return_mode: sum-->sum(), mean-->mean, vector --> vector
        '''
        # inference and discriminative part
        s, mask, y = incomings
        ebd, prob_y, a_mu, a_var, a, z_mu, z_var, z = self._inference(incomings, dev_stage=dev_stage)
        # generate initial state for sc-lstm
        u_mu, u_var, u, hid = self._gen_lm_init([y,z], dev_stage=dev_stage)
        # shift and drop words
        ebd_shift, word_drop_num = self._word_dropout(ebd, self.keep_rate, deterministic=dev_stage)
        ebd_shift = self._shift(ebd_shift)
        # lm
        cell_init = T.zeros(shape=hid.shape, dtype=theano.config.floatX) #initial value for cell
        dec_out = get_output(self.l_lm, {self.l_dec_cell_in:cell_init,self.l_label_in:y, self.l_dec_hid_in:hid,
                            self.l_dec_input_word_in:ebd_shift, self.l_mask_in:mask},
                           deterministic=dev_stage) #(batch_size*sent_length, word_dict_size)
        prob_x = get_output(self.l_recons_x, {self.l_dec_out_in:dec_out},
                           deterministic=dev_stage)
        '''
        labeled data loss lower bound: log(p(x,y)) >= \int_a&z{ q(a,z|x,y) log[p(x|u,yz)p(u|yz)p0(z)p0(y)/q(a,z|x,y)]}
        for reconstruction loss term, we simply use 1 MC sample:
            l_recons = -\int_a&z{ q(a,z|x,y)log[p(x|uyz)p(u|yz)] } = -log[ p(x|u,y,z)] = cross entropy loss(recons_x , s)
        for KL loss term,
            -\int_a&z{q(a|x)q(z|ayx) log[p0(z)] = -\int_z{q(z|ayx)log[p0(z)]}, while q(z|ayx)=N(z|z_mu, z_var), p0(z)=N(z|0,I)
            -\int_a&z{q(a|x)q(z|ayx) log[p0(y)] = -log[p0(y)]*1, while p0(y) = -Cat(y_label_in | y_i = 1/c)
            -\int_a&z{q(a|x)q(z|ayx) log[p(a|yz)] = -\int_z{q(a|x)log[p(a|yz)]}, while p(a|yz)=p(u|yz)
            +\int_a&z{q(a|x)q(z|ayx) log[q(a|x)]= +\int_a{q(a|x)log[q(a|x)]}, while q(a|x)=N(a|a_mu, a_var)
            +\int_a&z{q(a|x)q(z|axy) log[q(z|axy)] = +\int_z{q(z|axy) log[q(z|axy)]}, while q(z|axy)=N(z|z_mu, z_var)
        besides, we usually add classifier loss term,
            l_classifier = cross entropy loss(prob_y, y_label_in)
        '''
        # classifier loss
        loss_classifier = categorical_crossentropy(prob_y, y) #vector - batch_size
        acc = categorical_accuracy(prob_y, y) #\sum{ y*log(prob_y) }
        acc = T.mean(acc.astype(theano.config.floatX))
        # reconstruction loss
        loss_recons = categorical_crossentropy(prob_x, s.flatten())
        loss_recons = (loss_recons.reshape((-1,mask.shape[1])) * mask).sum(-1) # matrix batch_size*sent_length --> vector batch_size
        # kl loss
        kl_p0y, kl_p0z, kl_qz, kl_qa, kl_pa = self._calc_kl([y, z_mu, z_var, a_mu, a_var, u_mu, u_var],
                                                            return_mode='by_batch')
        loss_kl = kl_p0y - kl_p0z + kl_qz + kl_qa - kl_pa
        #ppl index
        valid_words = T.sum( mask ).astype('int64')
        batch_ppl = T.exp( T.sum( loss_recons) / valid_words.astype(theano.config.floatX))

        if return_mode == 'sum':
            return T.sum(loss_recons), T.sum(loss_kl), valid_words, word_drop_num, T.sum(loss_classifier), batch_ppl, acc
        elif return_mode == 'mean':
            return T.mean(loss_recons), T.mean(loss_kl), valid_words, word_drop_num, T.mean(loss_classifier), batch_ppl, acc
        else: # all vectors
            return loss_recons, loss_kl, valid_words, word_drop_num, loss_classifier, batch_ppl, acc

    def cost_unlabel(self, incomings, dev_stage = False, sample_by_prob = False ):
        s, mask = incomings
        prob_y = get_output(self.l_y,
                          {self.l_sents_in:s, self.l_mask_in:mask},
                          deterministic= dev_stage)
        '''
        sample_by_prob = False
        unlabeled data loss lower bound: log(p(x)) >= \int_ayz{ q(a,z,y|x) log[p(x|uyz)p(u|yz)p0(z)p0(y) / q(a,z,y|x)]}
                                                  = \int_ayz{ q(a|x)q(y|ax)q(z|ayx) log[p(x|uyz)p(u|yz)p0(z)p0(y) / q(a|x)q(y|ax)q(z|ayx)]}
        = \int_ayz{ q(y|ax)*q(a|x)q(z|ayx) log[p(x|uyz)p(u|yz)p0(z)p0(y) / q(a|x)q(z|ayx)}
          -\int_ayz{ q(a|x)q(y|ax)q(z|ayx) log[q(y|ax)]}
        = \sum_{y=c1}^{y=ck}{ q(y=ci|ax)* \int_az{ q(a|x)q(z|a,y=ci,x) log[p(x|u,y=ci,z)p(u|y=ci,z)p0(z)p0(y=ci) / q(a|x)q(z|a,y=ci,x)]}}
          -\int_y{q(y|ax)log[q(y|ax)]}
        = \sum_{y=c1}^{y=ck}{ q(y=ci|ax) * -Loss_Labeled(x, y=ci) + Entropy(q(y|ax))
        Loss: \sum_{y=c1}^{y=ck}{ q(y=ci|ax) * Loss_Labeled(x, y=ci) - Entropy(q(y|ax))

        sample_by_prob = True: sample a label from prob_y
        '''
        loss_entropy = categorical_crossentropy(prob_y,prob_y)
        loss_recons, loss_kl = T.zeros((s.shape[0],)), T.zeros((s.shape[0],))
        if sample_by_prob:  # hard sampled loss according to the prob of y
            print 'yi is sampled from prob_y'
            _srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            s_prob = _srng.uniform((s.shape[0],),dtype=theano.config.floatX)
            yi = T.eye(self.dimy)[T.switch(s_prob <= y_prob[:,0],0,1)] # one-hot
            loss_recons, loss_kl, _,_,_ ,_,_= self.cost_label([s,mask,yi],dev_stage=dev_stage, return_mode='vector')
        else:  # soft weighted loss
            print 'calc all yi in prob_y'
            for i in range(self.dimy):
                yi = T.zeros(prob_y.shape,dtype=theano.config.floatX)
                yi = T.set_subtensor(yi[:,i], 1.0)
                loss_recons_yi, loss_kl_yi, _,_,_ ,_,_= self.cost_label([s,mask,yi],dev_stage=dev_stage, return_mode='vector')
                loss_recons += loss_recons_yi * prob_y[:,i]
                loss_kl += loss_kl_yi * prob_y[:,i]

        valid_words = T.sum(mask).astype('int64')
        batch_ppl = T.exp( T.sum( loss_recons) / valid_words.astype(theano.config.floatX))
        return T.mean(loss_recons), T.mean(loss_kl), valid_words, T.mean(loss_entropy), batch_ppl

    def train_function(self, semi_supervised= True, unlabel_stable=False):
        '''
        use_unlabel == True, semi-superviesd learning
        return: train function for 1 epoch use
        '''
        self.semi_supervised = semi_supervised
        sym_klw = T.scalar('sym_klw',dtype=theano.config.floatX) # symbolic scalar of warming up
        sym_cw = T.scalar('sym_cw',dtype=theano.config.floatX) # classifier warm up
        sym_s = T.matrix('sym_s',dtype='int64')
        sym_mask = T.matrix('sym_mask',dtype=theano.config.floatX)
        sym_y = T.matrix('sym_label',dtype=theano.config.floatX)
        sym_s_u = T.matrix('sym_s_u',dtype='int64')
        sym_mask_u = T.matrix('sym_mask_u', dtype=theano.config.floatX)
        num_l, num_u = sym_s.shape[0].astype(theano.config.floatX), 0.0
        if self.semi_supervised:
            print 'Train with unlabel data.'
            num_u = sym_s_u.shape[0].astype(theano.config.floatX)
        #get labeled/unlabeled cost
        outs1 = self.cost_label([sym_s, sym_mask, sym_y], dev_stage=False, return_mode = 'mean')
        loss_recons, loss_kl, valid_words, word_drop_num, loss_classifier, batch_ppl, acc = outs1
        loss_recons_u, loss_kl_u,loss_entropy_u, batch_ppl_u = 0.0,0.0,0.0,0.0
        valid_words_u = 0
        if self.semi_supervised:
            outs2 = self.cost_unlabel([sym_s_u, sym_mask_u], dev_stage=unlabel_stable, sample_by_prob=self.sample_unlabel)
            loss_recons_u, loss_kl_u, valid_words_u, loss_entropy_u, batch_ppl_u = outs2
        '''
        total Loss:
        L = Loss_labeled(s,mask,y) + beta*(n_l+n_u)/n_l * Loss_classisifer(s,mask,y)
            + Loss_unlabel(s_u, mask_u)
        L = recons_term + sym_klw_term + loss_classifier_term - loss_entropy_u
        '''
        alpha = sym_cw * self.cost_beta * ( num_l + num_u ) / num_l
        total_cost = loss_recons * num_l + loss_recons_u * num_u\
                     + sym_klw * ( loss_kl * num_l + loss_kl_u * num_u)\
                     + alpha * loss_classifier * num_l\
                     - loss_entropy_u * num_u
        total_cost /= (num_l + num_u)
        train_params = self.get_params(only_trainable=True)
        all_grads = theano.grad(total_cost,train_params)
        all_grads = [T.clip(g, -self.grad_clipping, self.grad_clipping) for g in all_grads]
        all_grads = total_norm_constraint( all_grads, max_norm=self.max_norm )
        #all_grads = [T.clip(g, -self.grad_clipping, self.grad_clipping) for g in all_grads]
        updates = adam(all_grads,train_params, self.lr, self.beta1, self.beta2)
        if self.semi_supervised:
            train_input = [sym_s, sym_mask, sym_y, sym_s_u, sym_mask_u, sym_klw, sym_cw]
            train_output = [total_cost,
                            loss_recons, loss_recons_u, loss_kl, loss_kl_u, alpha, loss_classifier, loss_entropy_u,
                            batch_ppl, batch_ppl_u, valid_words, valid_words_u, word_drop_num, acc]
        else:
            train_input = [sym_s, sym_mask, sym_y, sym_klw, sym_cw]
            train_output = [total_cost, loss_recons, loss_kl, loss_classifier,
                            batch_ppl, valid_words, word_drop_num, acc]
        train_f = theano.function(inputs=train_input, outputs=train_output,updates=updates, name='train_function')
        return train_f

    def eval_function(self):
        '''
        only for labeled data
        '''
        print 'Evaluation function.'
        sym_s = T.matrix('sym_s',dtype='int64')
        sym_mask = T.matrix('sym_mask',dtype=theano.config.floatX)
        sym_y = T.matrix('sym_label',dtype=theano.config.floatX)

        outs1 = self.cost_label([sym_s, sym_mask, sym_y], dev_stage= True, return_mode = 'mean')
        loss_recons, loss_kl, valid_words, word_drop_num, loss_classifier, batch_ppl, acc = outs1
        total_cost = loss_recons + loss_kl + self.cost_beta * loss_classifier

        eval_input = [sym_s, sym_mask, sym_y]
        eval_output = [total_cost, loss_recons, loss_kl, loss_classifier,
                        batch_ppl, valid_words, acc]
        eval_f = theano.function(inputs=eval_input,outputs= eval_output,name='eval_function')
        return eval_f

    def data_analysis_function(self):
        '''
        After the training, use this function to show some results
        '''
        sym_s = T.matrix('sym_s',dtype='int64')
        sym_mask = T.matrix('sym_mask',dtype=theano.config.floatX)
        sym_y = T.matrix('sym_label',dtype=theano.config.floatX)

        _,_, a_mu, a_var, a, z_mu, z_var, z = self._inference([sym_s, sym_mask, sym_y], dev_stage=True)
        u_mu, u_var, u,_ = self._gen_lm_init([sym_y, z], dev_stage=True)
        kl_p0z_by_unit, kl_qz_by_unit, kl_qa_by_unit, kl_pa_by_unit =\
            self._calc_kl([sym_y, z_mu, z_var, a_mu, a_var, u_mu, u_var], return_mode='by_unit')

        ana_input = [sym_s, sym_mask, sym_y]
        ana_output = [a_mu, a_var, a, z_mu, z_var, z, u_mu, u_var, u,
                     kl_p0z_by_unit, kl_qz_by_unit, kl_qa_by_unit, kl_pa_by_unit]
        ana_f = theano.function(inputs=ana_input,outputs= ana_output,name='data_analysis_function')
        return ana_f

    def dec_hid_function(self):
        '''
        :returns a theano function which compute the initial hidden state of dec LSTM
        the function input: sym_y and sym_z, first reshaped to (batch_size, dimy) (batch_size, dimz)
                     output hid_init, whose shape is (batch_size, dec_num_units)
        '''
        print 'building beam search prepare function 01 ...'
        sym_y, sym_z = T.matrix('beam_search_label',dtype=theano.config.floatX), T.matrix('beam_search_z',dtype=theano.config.floatX)

        hid_init = get_output(self.l_hid, {self.l_z_in:sym_z, self.l_label_in:sym_y}, deterministic=True)#(batch_size,dec_num_unit)
        dec_hid_input = [sym_y, sym_z]
        dec_hid_output = [hid_init]
        dec_hid_f = theano.function(inputs=dec_hid_input, outputs=dec_hid_output, name='dec_hid_init_function')
        return  dec_hid_f

    def dec_step(self, word_init_stage = False):
        '''
        :param word_init_stage == True, means the initialization stage of beam search, all word ebeddings should set to zeros.
                                  else, use self.l_ebd
        :return: step function of dec lstm, which is used in beam search
        the function input: sym_cell_pre, sym_hid_pre(beam_size,dec_num_units), sym_word_pre(beam_size,)(vector)
        '''
        sym_cell_pre = T.matrix('lstm_step_cell',dtype=theano.config.floatX)
        sym_hid_pre = T.matrix('lstm_step_hid', dtype=theano.config.floatX)
        sym_da_pre =  T.matrix('lstm_step_da',dtype=theano.config.floatX)
        sym_word_pre = T.vector('lstm_step_word_idx', dtype='int64')
        # first convert index of words to embeddings, note that during initialization, word_pre = -1
        if word_init_stage:
            w_ebd = T.zeros((sym_word_pre.shape[0], self.word_ebd_dims),dtype=theano.config.floatX)
        else:
            w_ebd = get_output(self.l_ebd,{self.l_sents_in:sym_word_pre},deterministic=True)

        cell, hid, da = self.l_lm.one_step(w_ebd, sym_cell_pre, sym_hid_pre, sym_da_pre)

        prob_x = get_output(self.l_recons_x, {self.l_dec_out_in:hid},
                            deterministic=True) #(batch_size*1,word_dict_size) no need to reshape
        dec_step_input = [sym_word_pre, sym_cell_pre, sym_hid_pre, sym_da_pre]
        dec_step_output = [prob_x, cell, hid, da]
        dec_step_f = theano.function(inputs=dec_step_input,outputs=dec_step_output,name='dec_step_function')
        return dec_step_f

    def beam_search(self, y, z, beam_size, max_sent_length = 100):
        '''
        generating sentences form language model employing Beam Search,
        code reference to dl4mt. ---> https://github.com/kyunghyuncho/dl4mt-material/blob/master/session3/nmt.py
        y,z: samples of label and hidden state (batch_size * dimy) (batch_size * dimz)
            usually the batch_size == 1
        beam_size: beam search params, [5, 30]
        init_fun: calc the initial hidden state of the language model
        step_fun: calc each lstm step of the language model, compile beyond this function
        '''
        init_fun = self.dec_hid_function()
        init_step_fun = self.dec_step(word_init_stage=True)
        step_fun = self.dec_step(word_init_stage=False)
        live_k, dead_k = 1, 0
        sample, sample_score = [], []
        hyp_samples, hyp_score, hyp_hid, hyp_cell = [[]] * live_k, np.zeros(live_k).astype('float32'), [], []
        #init of sampler
        hid_init = init_fun(y,z)  #(1, dec_num_units)
        hid_pre = np.tile(hid_init[0], (live_k, 1))
        cell_pre = np.zeros((live_k, self.dec_num_units), dtype='float32')
        next_w = -1 * np.ones((live_k,)).astype('int64')

        for ii in range(max_sent_length):
            da = np.tile(y, (cell_pre.shape[0], 1))
            print da.shape
            print cell_pre.shape
            print hid_pre.shape
            if ii > 0:
                prob_words, cell, hid, _ = step_fun(next_w, cell_pre, hid_pre, da)
            else:
                prob_words, cell, hid, _ = init_step_fun(next_w, cell_pre, hid_pre, da)

            cand_scores = hyp_score[:,None] - np.log(prob_words)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(beam_size - dead_k)]

            trans_idx = ranks_flat / np.int64(self.word_dict_size) # sample index (floor function)
            word_idx = ranks_flat % np.int64(self.word_dict_size)  # word index (mod)
            costs = cand_flat[ranks_flat]

            new_hyp_samples, new_hyp_scores, new_hyp_hid, new_hyp_cell = [], np.zeros(beam_size-dead_k).astype('float32'), [], []
            for idx, [ti, wi] in enumerate(zip( trans_idx, word_idx)):
                new_hyp_samples.append( hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_hid.append(copy.copy(hid[ti]))
                new_hyp_cell.append(copy.copy(cell[ti]))

            new_live_k, hyp_samples, hyp_score, hyp_hid, hyp_cell = 0, [], [], [], []
            for idx in range(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0: # which means reach the end tag of the sentence <EOS>, its word embedding index is 0.
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else: # the sentence not terminated, needs further calculation
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_score.append(new_hyp_scores[idx])
                    hyp_hid.append(new_hyp_hid[idx])
                    hyp_cell.append(new_hyp_cell[idx])
            #
            hyp_score = np.array(hyp_score)
            live_k = new_live_k
            if new_live_k < 1:
                break

            if dead_k >= beam_size:
                break

            next_w, hid_pre, cell_pre = np.array([w[-1] for w in hyp_samples]), np.array(hyp_hid) , np.array(hyp_cell)
        #end for
        if live_k > 0:
            for idx in range(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_score[idx])
        #
        return sample, sample_score


    def load_pretrain(self, pretrain_path):
        # the loaded data has certain format
        import cPickle as pkl
        with open(pretrain_path, 'rb') as f:
            load_wemb = pkl.load(f)
            wemb = self.l_ebd.get_params()
            for i in range(len(load_wemb)):
                #print(load_wemb[i].shape)
                #print(wemb[i].get_value().shape)
                wemb[i].set_value(load_wemb[i])

            load_lstm = pkl.load(f)
            lstm = self.l_c0.get_params()
            for i in range(len(load_lstm)):
                #print(lstm[i].get_value().shape)
                #print(load_lstm[i].shape)
                lstm[i].set_value(load_lstm[i])
