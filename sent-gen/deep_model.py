import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np
import lasagne
import copy
import lasagne.nonlinearities as nonlinearities
import lasagne.init as init
from lasagne.layers.base import *
from theano.tensor.shared_randomstreams import RandomStreams

def _lstm(in_gates, hid_pre, cell_pre, mask,
           W_gates , W_hid_to_gates, W_cell_to_gates, b_gates,
           # model parameters
           num_units,
           act_ingate , act_forgetgate , act_modulationgate, act_outgate, act_out,
           use_peepholes = False, grad_clip_vals_in = None
           ):
    '''
    LSTM step Gate names are taken from http://arxiv.org/abs/1409.2329 figure 1
    :param in_gates (batch_size , z1) ,  hid_pre (batch_size , z2)
    :param W_gates ( z1 , num_units*4) , W_hid_to_gates ( z2 , num_units*4) , b_gates ( 1 , num_units*4)
    :param cell_previous, W_cell_to_gates : lstm parameters
    :param ..., act_outgate, act_out : active functions

    gates: inputs for lstm layers, gates = dot( in_gates, W_gates) + b_enc_gates + dot( hid_pre , W_hid_to_gates)
                                          matrix size = (batch_size, z),( z , num_units*4) , (1 , num_units*4)
    '''
    def slice_w(x, n):
        return x[:, n*num_units:(n+1)*num_units]

    def slice_c(x, n):
        return x[n*num_units:(n+1)*num_units]

    def clip(x):
        return theano.gradient.grad_clip(x, grad_clip_vals_in[0], grad_clip_vals_in[1])

    #calc gates
    gates = T.dot(in_gates, W_gates) + b_gates + T.dot(hid_pre, W_hid_to_gates)

    ingate = slice_w(gates, 0)
    forgetgate = slice_w(gates, 1)
    modulationgate = slice_w(gates, 2)
    outgate = slice_w(gates, 3)

    if use_peepholes:
        ingate += cell_pre*slice_c(W_cell_to_gates, 0)
        forgetgate += cell_pre*slice_c(W_cell_to_gates, 1)

    if grad_clip_vals_in is not None:
        print('STEP: CLipping gradients IN', grad_clip_vals_in)
        ingate = clip(ingate)
        forgetgate = clip(forgetgate)
        modulationgate = clip(modulationgate)

    ingate = act_ingate(ingate)
    forgetgate = act_forgetgate(forgetgate)
    modulationgate = act_modulationgate(modulationgate)

    if grad_clip_vals_in is not None:
        ingate = clip(ingate)
        forgetgate = clip(forgetgate)
        modulationgate = clip(modulationgate)

    cell = forgetgate*cell_pre + ingate*modulationgate

    if use_peepholes:
        outgate += cell*slice_c(W_cell_to_gates, 2)

    if grad_clip_vals_in is not None:
        outgate = clip(outgate)

    outgate = act_outgate(outgate)

    if grad_clip_vals_in is not None:
        outgate = clip(outgate)

    hid = outgate*act_out(cell)
    #return [cell,hid]
    return [ mask * cell + (1-mask) * cell_pre, mask * hid + (1-mask) * hid_pre ]

__all__ = [ "SentGenLayer" ]
class SentGenLayer( Layer ):

    def __init__(self,
                 #Model Params
                 input_layer_sentences,
                 batch_size, dimz, max_sen_length, enc_num_units, dec_num_units,
                 word_ebds, num_all_words, ebd_dim,
                 #Default Params -- y
                 use_y = False, dimy = 0,
                 #Other Params -- active functions
                 act_ingate = nonlinearities.sigmoid, act_forgetgate = nonlinearities.sigmoid,
                 act_modulationgate = nonlinearities.tanh, act_outgate = nonlinearities.sigmoid, act_out = nonlinearities.tanh,
                 #Default Params -- model params
                 use_fixed_ebd = False,
                 use_peepholes = False,
                 grad_clip_vals = [-10.0, 10.0]
                 ):
        #initial state
        ini_normal = init.Normal(std=5e-2 , mean=0.0)
        ini_zero = init.Uniform(range=1e-8, mean=0.0)
        ini_one = init.Constant(1.0)
        # Initialize parent layer
        super(SentGenLayer, self).__init__(input_layer_sentences)
        # init hyperparams and embedding matrix
        self.batch_size = batch_size
        self.dimz = dimz
        self.use_y = use_y
        self.dimy = dimy
        self.max_sen_length = max_sen_length
        self.enc_num_units = enc_num_units
        self.dec_num_units = dec_num_units
        self.num_all_words = num_all_words
        self.ebd_dim = ebd_dim
        self.use_fixed_ebd = use_fixed_ebd
        self.use_peepholes = use_peepholes
        self.grad_clip_vals = grad_clip_vals

        # init active functions
        self.act_ingate = act_ingate
        self.act_forgetgate = act_forgetgate
        self.act_modulationgate = act_modulationgate
        self.act_outgate = act_outgate
        self.act_out = act_out

        # init weights of enc lstm 3
        self.enc_b_gates = self.add_param( ini_zero, [4*self.enc_num_units], name = "SentGen: Encoder Lstm: b_gates")
        self.enc_W_gates = self.add_param( ini_normal, [self.ebd_dim, 4*self.enc_num_units], name = "SentGen: Encoder Lstm: W_gates")
        self.enc_W_hid_to_gates = self.add_param( ini_normal, [self.enc_num_units, 4*self.enc_num_units], name = "SentGen: Encoder Lstm: W_hid_to_gates")
        # word ebds 1/0
        if not self.use_fixed_ebd: # not as a model parameters
            self.word_ebds = self.add_param( ini_normal, [self.num_all_words, self.ebd_dim], name = "SentGen: Word Embeddings: word_ebds")
        else:
            self.word_ebds = self.add_param( word_ebds.astype(theano.config.floatX), [self.num_all_words, self.ebd_dim], name = "SentGen: Word Embeddings: word_ebds")
        # varitional weights 4
        self.b_enc_to_mu = self.add_param( ini_zero, [self.dimz], name= "SentGen: Gaussian Linear Layer: b_enc_to_mu" )
        self.b_enc_to_sigma = self.add_param( ini_zero, [self.dimz], "SentGen: Gaussian Linear Layer: b_enc_to_sigma")
        if self.use_y:
            self.W_enc_to_mu = self.add_param( ini_normal, [self.enc_num_units + self.dimy, self.dimz], name = "SentGen: Gaussian Linear Layer: W_enc+Y_to_mu")
            self.W_enc_to_sigma = self.add_param( ini_normal, [self.enc_num_units + self.dimy, self.dimz], "SentGen: Gaussian Linear Layer: W_enc+Y_to_sigma")
        else:
            self.W_enc_to_mu = self.add_param( ini_normal, [self.enc_num_units, self.dimz], name = "SentGen: Gaussian Linear Layer: W_enc_to_mu")
            self.W_enc_to_sigma = self.add_param( ini_normal, [self.enc_num_units, self.dimz], "SentGen: Gaussian Linear Layer: W_enc_to_sigma")
        # init z to decoder 1
        if self.use_y:
            self.W_z_to_lm = self.add_param( ini_normal, [self.dimz + self.dimy, self.dec_num_units], "SentGen: Z: W_z+Y_to_lm")
        else:
            self.W_z_to_lm = self.add_param( ini_normal, [self.dimz, self.dec_num_units], "SentGen: Z: W_z_to_lm")
        # init weights of dec lstm 5
        self.dec_b_gates = self.add_param( ini_zero, [4*self.dec_num_units], name = "SentGen: Decoder Lstm: b_gates" )
        self.dec_W_gates = self.add_param( ini_normal, [self.ebd_dim, 4*self.dec_num_units], name = "SentGen: Decoder Lstm: W_gates")
        self.dec_W_hid_to_gates = self.add_param( ini_normal, [self.dec_num_units, 4*self.dec_num_units], name = "SentGen: Decoder Lstm: W_hid_to_gates")
        self.dec_W_lm_out = self.add_param( ini_normal, [self.dec_num_units, self.num_all_words], name = "SentGen: Decoder Prob Softmax Layer: W_lm_out")
        self.dec_b_lm_out = self.add_param( ini_zero, [self.num_all_words], name = "SentGen: Decoder Prob Softmax Layer: b_lm_out")

        # init if use peepholes 2/0
        if self.use_peepholes:
            self.enc_W_cell_to_gates = self.add_param( ini_normal, [3*self.enc_num_units], name = "SentGen: Encoder Lstm Peepholes: W_cell_to_gates")
            self.dec_W_cell_to_gates = self.add_param( ini_normal, [3*self.dec_num_units], name = "SentGen: Decoder Lstm Peepholes: W_cell_to_gates")
        else:
            self.enc_W_cell_to_gates = []
            self.dec_W_cell_to_gates = []

        # Setup initial values for hidden and cell of lstm
        self.enc_hid_init = T.zeros((self.batch_size, self.enc_num_units), theano.config.floatX)
        self.enc_cell_init = T.zeros((self.batch_size, self.enc_num_units), theano.config.floatX)
        self.dec_hid_init = T.zeros((self.batch_size, self.dec_num_units), theano.config.floatX)
        self.dec_cell_init = T.zeros((self.batch_size, self.dec_num_units), theano.config.floatX)
        self.dec_prob_words_init = T.zeros( (self.batch_size, self.num_all_words), theano.config.floatX) #big matrix
        self.cross_ent_init = T.zeros( (self.batch_size,), theano.config.floatX) #big matrix

    def get_params(self):
        # 13 fixed params + 2, the last one is word_ebds
        params = [self.enc_b_gates, self.enc_W_gates, self.enc_W_hid_to_gates,
                  self.dec_b_gates, self.dec_W_gates, self.dec_W_hid_to_gates, self.dec_b_lm_out, self.dec_W_lm_out,
                  self.W_enc_to_mu, self.b_enc_to_mu, self.W_enc_to_sigma, self.b_enc_to_sigma, self.W_z_to_lm]
        # peepholes and word embeddings
        if self.use_peepholes:
            params.extend([self.enc_W_cell_to_gates, self.dec_W_cell_to_gates ])
        #
        params.extend([self.word_ebds])

        return params

    def get_cost(self, keep_rate, drop_out, #inputs
                 s, mask, kl_weight, y = None, *args, **kwargs): #givens
        '''
        MAKE SURE: dtype = theano.config.floatX ( default float32 )
        Cost functions, s is one-hot matrix represents sentence batches with uncertain lengths, mask is a flag that indicates the lengths
        we assume input s is with size ( max_length , batch_size ) , mask has size ( max_length , batch_size)
                        guass_rng: sample from N(0,1), size ( batch_size , dimz)
        ABOUT word embeddings: dict[0] = <eos> dict[1]=<unk> randomly generated vectors
        y: (batch_size, dimy) one-hot
        '''
        if y is None and self.use_y is True:
            raise ValueError('Y must be given.')

        srng = RandomStreams( )
        def _rnn_drop_out(y, yshape, rescale = True):
            '''
            LSTM dropout trick, only training
            ref: Recurrent Neural Network Regularization, Wojciech Zaremba, ICLR 2015
            y: layer that need dropout
            drop_rate: [0,1)
            '''
            if 0 < drop_out < 1:
                retain_prob = 1 - drop_out
                amp = 1.0
                if rescale:
                    amp = retain_prob
                drop_out_mask = srng.binomial(yshape, n=1, p= retain_prob, dtype=theano.config.floatX) / amp
                return y * drop_out_mask
            else:
                return y

        def _word_drop_out(x, mask):
            '''
            #word dropout using switch  use self.word_ebds[1] UNK? OR zero? now we choose (1-keep_rate)% of all the positions
            #if 0 < keep_rate < 1 random sampled mask;
            #if keep_rate >=1  mask = ones;
            #if keep_rate <=0 mask = zeros;
            '''
            if 0 < keep_rate < 1:
                #keep_rate - T.floor(keep_rate) make sure 0=<p<=1
                keep_position = srng.binomial((self.max_sen_length ,self.batch_size), n=1, p= keep_rate - T.floor(keep_rate), dtype=theano.config.floatX)
                keep_mask = keep_position.repeat( self.ebd_dim ).reshape((self.max_sen_length, self.batch_size, self.ebd_dim ))
                return [x * keep_mask, T.sum(mask*(1.0 - keep_position)).astype('int64')]
            elif keep_rate <= 0:
                return [T.zeros((self.max_sen_length, self.batch_size, self.ebd_dim ), dtype=theano.config.floatX),\
                        T.sum(mask).astype('int64') ]
            else:
                return [x, T.sum(mask*0.0).astype('int64') ]

        def _lstm_decode_step( x, mask, s, hid_drop_mask,
                           # Reucrent parameters
                           hid_pre , cell_pre, cross_entropy_prob_pre ,
                           #weights
                           W_gates , W_hid_to_gates, W_cell_to_gates, W_lm_out, b_gates, b_lm_out
                           ):
            '''
            :param mask:  mask for rnns, (batch_size , 1) slice
            :param hid_pre , cell_pre , word_idx_pre:  recurrent parameters, word_idx_pre ( batch_size , num_words)
            :param softmax_output_pre: outputs
            :param W_gates , W_hid_to_gates , W_cell_to_gates , W_lm_out
            :param b_extra: z for init b_extra = dot( z , W_z_to_lm )
            '''
            #b_gates += b_extra
            r_mask = mask.repeat(self.dec_num_units).reshape((-1,self.dec_num_units)) #repeat each row from <batch_size> to <batch_size , num_units>
            cell, hid = _lstm( x , hid_pre , cell_pre, r_mask,
                               W_gates , W_hid_to_gates, W_cell_to_gates, b_gates,
                               self.dec_num_units,
                               self.act_ingate, self.act_forgetgate, self.act_modulationgate, self.act_outgate, self.act_out,
                               self.use_peepholes, self.grad_clip_vals)
            #TO DO: softmax can be speed up by hiretical model
            prob_words = T.nnet.softmax( T.dot( hid * hid_drop_mask, W_lm_out ) + b_lm_out )
            s_flat = s.flatten()
            s_flat_idx = T.arange(self.batch_size) * self.num_all_words + s_flat
            prob_words_flat = prob_words.flatten()
            cross_entropy_prob = prob_words_flat[s_flat_idx]
            return [ hid , cell , cross_entropy_prob]

        def _lstm_encode_step(x, mask,
                              # recurrent params
                              hid_pre , cell_pre ,
                              # weights
                              W_gates , W_hid_to_gates, W_cell_to_gates, b_gates):
            '''
            :param mask and x are sliced data for theano loops;  x : word embeddings ( batch_size , ebd_dims)
            :param hid_pre ,cell_pre: lstm recurrent params
            :param W_gates ,W_hid_to_gates , W_cell_to_gates ,b_gates: lstm params
            '''
            r_mask = mask.repeat(self.enc_num_units).reshape((-1,self.enc_num_units))
            cell , hid = _lstm( x, hid_pre, cell_pre , r_mask,
                                W_gates, W_hid_to_gates, W_cell_to_gates, b_gates,
                                self.enc_num_units,
                                self.act_ingate , self.act_forgetgate, self.act_modulationgate, self.act_outgate, self.act_out,
                                self.use_peepholes, self.grad_clip_vals)

            return [ hid , cell ]

        guass_rng = srng.normal((self.batch_size, self.dimz)).astype( theano.config.floatX )
        # first change one-hot sentences to word embeddings
        s_flat = s.flatten()
        x_flat = self.word_ebds[s_flat]
        x = x_flat.reshape((self.max_sen_length, self.batch_size,  self.ebd_dim ))

        # encode rnns
        enc_seqs = [ _rnn_drop_out(x,(self.max_sen_length, self.batch_size,  self.ebd_dim )),  mask ] # shape[0] = max_length
        enc_init = [ self.enc_hid_init , self.enc_cell_init ] # make sure dtype = theano.config.foloatX
        enc_nonseqs = [ self.enc_W_gates , self.enc_W_hid_to_gates, self.enc_W_cell_to_gates, self.enc_b_gates ]
        enc_output = theano.scan( _lstm_encode_step,
                                  sequences = enc_seqs, outputs_info = enc_init , non_sequences = enc_nonseqs,
                                  go_backwards = False )[0] #[hid , cell] each with size ( max_length , batch_size , enc_num_units
        enc_out = _rnn_drop_out( enc_output[0][-1], (self.batch_size, self.enc_num_units))  #take the last one, ( batch_size , enc_num_units )
        if self.use_y:
            enc_out = T.concatenate([enc_out, y], axis=1)

        #TODO Gaussian sample tricks
        # mean and sigma of z
        self.mu_z = T.dot(enc_out, self.W_enc_to_mu ) + self.b_enc_to_mu # ( batch_size , dimz)
        self.log_sigma_z = 0.5*(T.dot(enc_out, self.W_enc_to_sigma) + self.b_enc_to_sigma)
        # sample z
        self.z = self.mu_z + T.exp( self.log_sigma_z )*guass_rng  #(batch_size , dimz)

        #decode lm
        if self.use_y:
            self.z = T.concatenate([self.z, y] , axis=1 )

        self.extra_hid_init = T.dot( self.z , self.W_z_to_lm ) # z to lm
        #word dropout
        x, self.word_dropout_num = _word_drop_out(x, mask)

        #shift the x so that the first one is "START" (all zero), while mask do not need change
        x_shift = T.zeros_like( x )
        x_shift = T.set_subtensor(x_shift[1:] , x[:-1] ) # to right, the last word is not needed
        #drop out can't use in scan
        dec_hid_drop_mask = T.ones((self.max_sen_length, self.batch_size, self.dec_num_units), dtype = theano.config.floatX )
        dec_hid_drop_mask = _rnn_drop_out(dec_hid_drop_mask,(self.max_sen_length, self.batch_size, self.dec_num_units))
        dec_seqs = [ _rnn_drop_out(x_shift,(self.max_sen_length, self.batch_size,  self.ebd_dim )), mask , s, dec_hid_drop_mask]
        #init, note that z is used here
        dec_init = [ self.dec_hid_init + self.extra_hid_init , self.dec_cell_init , self.cross_ent_init ]  # make sure dtype = theano.config.floatX
        dec_nonseqs = [ self.dec_W_gates , self.dec_W_hid_to_gates, self.dec_W_cell_to_gates, self.dec_W_lm_out, self.dec_b_gates ,self.dec_b_lm_out ]
        lm_out = theano.scan( _lstm_decode_step,
                              sequences = dec_seqs, outputs_info = dec_init , non_sequences = dec_nonseqs,
                              go_backwards = False )[0]
        _h, _c, prob_words_all = lm_out  # prob_words_all ( max_length , batch_size , num_words )
        #Loss function
        self.loss_recons = -T.log(1e-15 + prob_words_all) * mask #log likelihood
        self.word_num = T.sum(mask)
        self.ppl = T.exp( T.sum(self.loss_recons) / self.word_num) # ppl
        self.loss_kl =  0.5 * ( self.mu_z**2 + T.exp(self.log_sigma_z*2) - 1 - self.log_sigma_z*2)
        cost = ( T.sum( self.loss_recons ) + kl_weight * T.sum( self.loss_kl )) / self.batch_size
        return cost

    def get_train_results(self):
        return [T.sum(self.loss_kl)/self.batch_size, T.sum(self.loss_recons)/self.batch_size, self.ppl, self.word_num, self.word_dropout_num ]

    #TODO BEAM SEARCH WITH Y
    def gen_z_to_lm(self):
        '''
        Calculate initial state for z inputs
        '''
        z = T.vector('Beam_search_Z', dtype=theano.config.floatX)
        eout = T.dot( z , self.W_z_to_lm )
        print "Building generator 01..."
        f = theano.function([z],[eout])
        return f

    def gen_step(self):
        '''
        Decode step in beam search using _lstm function ( theano function)
        inputs( theano syms) w(batch_size,ebd_dim), hid_pre ,cell_pre, mask
        outputs probs_words, hid, cell
        params live_k is used instead of params batch_size
        '''
        w = T.vector('Beam_search_W_idx', dtype='int64') #word index (vector)
        hid_pre = T.matrix('Beam_search_hid_pre', dtype=theano.config.floatX)
        cell_pre = T.matrix('Beam_search_cell_pre', dtype=theano.config.floatX)
        mask = T.matrix('Beam_search_mask', dtype=theano.config.floatX)
        w_ebd = T.switch(w[:,None] < 0 , T.alloc(0.,1,self.ebd_dim) , self.word_ebds[w] )
        cell, hid = _lstm( w_ebd, hid_pre , cell_pre, mask,
                           self.dec_W_gates , self.dec_W_hid_to_gates, self.dec_W_cell_to_gates,  self.dec_b_gates,
                           self.dec_num_units,
                           self.act_ingate , self.act_forgetgate ,self.act_modulationgate, self.act_outgate, self.act_out,
                           self.use_peepholes, self.grad_clip_vals)
        prob_words = T.nnet.softmax( T.dot( hid , self.dec_W_lm_out, ) + self.dec_b_lm_out )
        inps = [w, hid_pre, cell_pre, mask]
        outs = [prob_words, hid , cell]
        print "Building generator 02..."
        f = theano.function(inps, outs, name='decoder_gen')
        return f


    def generator(self, z, beam_size, init_fun, step_fun):
        live_k, dead_k = 1, 0
        sample, sample_score = [], []
        hyp_samples, hyp_score, hyp_hid, hyp_cell = [[]] * live_k, np.zeros(live_k).astype('float32'), [], []
        #init of sampler
        extra_hid = init_fun(z)
        hid_pre = np.tile(extra_hid, (live_k, 1))
        cell_pre = np.zeros((live_k, self.dec_num_units), dtype='float32')
        next_w = -1 * np.ones((live_k,)).astype('int64')

        for ii in range(self.max_sen_length):
            one_mask = np.ones((live_k, self.dec_num_units), dtype='float32')
            prob_words, hid, cell = step_fun(next_w, hid_pre, cell_pre, one_mask)
            cand_scores = hyp_score[:,None] - np.log(prob_words)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(beam_size - dead_k)]

            trans_idx = ranks_flat / np.int64(self.num_all_words) # sample index (floor function)
            word_idx = ranks_flat % np.int64(self.num_all_words)  # word index (mod)
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
















