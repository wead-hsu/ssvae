import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.objectives import *
from lasagne.layers import *
from lasagne import init
from lasagne.nonlinearities import *
from lasagne.updates import total_norm_constraint
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.updates import adam
from layer.sample import *
from utils.VariationalCost import *
from layer.sclstm import *
from layer.meanlstm import *
import copy

# code by Haoze Sun 2016.7

class DeepModel():
    def __init__(self,
                #model params used in stage initialization
                num_units, dim_z, word_ebd_dims, word_dict_size, dim_y=2, drop_out=0.50, keep_rate=0.9,
                #train/evalue params that unchange during train
                use_baseline=False, grad_clipping=5.0, max_norm=20.0, alpha=2.0, lr=0.0004
                #other params that unchange
                ):
        print 'Sample model for M1+M2'
        self.num_units = num_units
        self.word_ebd_dims, self.word_dict_size = word_ebd_dims, word_dict_size
        self.dim_y, self.dim_z = dim_y, dim_z
        self.drop_out, self.keep_rate = drop_out, keep_rate
        self.use_baseline = use_baseline
        self.grad_clipping, self.max_norm = grad_clipping, max_norm
        self.alpha = alpha  # cost = L(x,y)+U(x)+\alpha*cost_classifier
        self.lr = lr

    def build_model(self, use_mean_lstm=False, act_fun=lasagne.nonlinearities.tanh,  # density layer active function
                    word_ebd_init=init.Normal(1e-2), b_init=init.Normal(1e-4), W_init=init.GlorotNormal()):
        # --------------------------------  Global Inputs  ------------------------------------------------------------
        self.l_sents_in = InputLayer((None, None))  # sentences inputs as word indexes.
        self.l_mask_in = InputLayer((None, None))
        self.l_label_in = InputLayer((None, self.dim_y))  # one hot
        # for unlabel data, y is generated from classifier,else y is a parameter in trainning

        # ---------------------------------  Word Embedding  ---------------------------------------------------------
        # ## Input Nodes: l_sents_in
        self.l_ebd = EmbeddingLayer(self.l_sents_in, self.word_dict_size, self.word_ebd_dims, W=word_ebd_init,
                                    name='EbdLayer')  # we do dropout later.
        self.l_enc_sents_in = InputLayer((None, None, self.word_ebd_dims))  # sentences inputs as for classifier and encoder
        self.l_dec_sents_in = InputLayer((None, None, self.word_ebd_dims))  # for decoder, shifted and word-dropoutted

        # ---------------------------------  Classifier ---------------------------------------------------------
        # ## Input Nodes: l_enc_sents_in, l_mask_in
        self.l_c_sents_drop = DropoutLayer(self.l_enc_sents_in, p=self.drop_out, name='Classifier Sents Dropout')
        if use_mean_lstm:  # we do dropout later for loading pretraining weights
            self.l_c = MeanLstmLayer(self.l_c_sents_drop, num_units=self.num_units, mask_input=self.l_mask_in,
                                      grad_clipping=self.grad_clipping, name='Classifier Mean')
        else:
            self.l_c = LSTMLayer(self.l_c_sents_drop, num_units=self.num_units, mask_input=self.l_mask_in,
                                  grad_clipping=self.grad_clipping, only_return_final=True, name='Classifier Final')
        self.l_c_drop = DropoutLayer(self.l_c, p=self.drop_out, name='Classifier LSTM Dropout')
        self.l_c_to_y = DropoutLayer(batch_norm(DenseLayer(self.l_c_drop, num_units=self.num_units,
                                                W=W_init, b=b_init, nonlinearity=act_fun, name='c_to_y'),
                                                name='c_to_y'), p=self.drop_out, name='c_to_y')
        self.l_y = DenseLayer(self.l_c_to_y, num_units=self.dim_y, W=W_init, b=b_init,
                              nonlinearity=softmax, name='y_pred')

        # ---------------------------------  Inference Network ---------------------------------------------------------
        # ## Input Nodes: l_enc_sents_in, l_label_in, l_mask_in
        self.l_enc_sents_drop = DropoutLayer(self.l_enc_sents_in, p=self.drop_out, name='Enc Sents Dropout')
        # Encoder LSTM
        self.l_x = DropoutLayer(LSTMLayer(self.l_enc_sents_drop, num_units=self.num_units, mask_input=self.l_mask_in,
                                          grad_clipping=self.grad_clipping, only_return_final=True,name='Enc LSTM'),
                                p=self.drop_out, name='Enc LSTM Drop')
        # Encoder Dense Layer(s), use a class if many
        self.l_x_to_a = DropoutLayer(batch_norm(DenseLayer(self.l_x, num_units=self.num_units, W=W_init, b=b_init,
                                                nonlinearity=act_fun, name='x_to_a'), name='x_to_a'),
                                     p=self.drop_out,name='x_to_a')
        # combine information from label and encoder
        self.l_label_to_enc = DropoutLayer(DenseLayer(self.l_label_in, num_units=self.num_units, W=W_init, b=b_init,
                                            nonlinearity=act_fun, name='label_to_enc'),
                                      p=self.drop_out, name='label_to_enc')
        self.l_xy = ConcatLayer([self.l_x_to_a, self.l_label_to_enc], axis=1, name='Concat_xy')
        self.l_xy = DropoutLayer(batch_norm(DenseLayer(self.l_xy, num_units=self.num_units, W=W_init, b=b_init,
                                            nonlinearity=act_fun, name='xy'), name='xy'),
                                      p=self.drop_out, name='xy')
        self.l_xy_to_z = DropoutLayer(batch_norm(DenseLayer(self.l_xy, num_units=self.dim_z, W=W_init, b=b_init,
                                                 nonlinearity=act_fun, name='xy_to_z'),name='xy_to_z'),
                                      p=self.drop_out, name='xy_to_z')
        # sample z
        self.l_z_mu = DenseLayer(self.l_xy_to_z, self.dim_z, W=W_init, b=b_init, nonlinearity=None, name='z_mu')
        self.l_z_var = DenseLayer(self.l_xy_to_z, self.dim_z, W=W_init, b=b_init, nonlinearity=None,name='z_var')
        self.l_z = SimpleSampleLayer(self.l_z_mu, self.l_z_var, name='z_sample')

        # ---------------------------------  Generation Network -------------------------------------------------------
        # ## Input Nodes: l_label_in, l_z_in, l_dec_sents_in, l_mask_in
        # In this model, there is no interface for beam search.
        self.l_z_in = InputLayer((None, self.dim_z))
        self.l_label_to_dec = DropoutLayer(DenseLayer(self.l_label_in, num_units=self.num_units, W=W_init, b=b_init,
                                                      nonlinearity=act_fun, name='label_to_dec'),
                                           p=self.drop_out, name='label_to_dec')
        self.l_yz = ConcatLayer([self.l_label_to_dec, self.l_z_in], axis=1, name='Concat_yz')
        # Decoder Dense Layer(s)
        self.l_yz = DropoutLayer(batch_norm(DenseLayer(self.l_yz, num_units=self.num_units, W=W_init, b= b_init,
                                            nonlinearity=act_fun, name='yz'), name='yz'),
                                 p=self.drop_out, name='yz')
        # the last layer has no dropout
        self.l_hid = batch_norm(DenseLayer(self.l_yz, num_units=self.num_units, W=W_init, b=b_init,
                                                      nonlinearity=act_fun, name='yz_to_hid'), name='yz_to_hid')
        # language model
        self.l_lm = ScLSTMLayer(incoming=self.l_dec_sents_in, num_units=self.num_units, da_init=self.l_label_in,
                                 hid_init=self.l_hid, mask_input=self.l_mask_in,
                                grad_clipping=self.grad_clipping, name='ScLSTMLayer')
        self.l_rec = DenseLayer(DropoutLayer(ReshapeLayer(self.l_lm, shape=(-1, self.num_units),
                                    name='ScLSTMLayer'), p=self.drop_out, name='ScLSTMLayer'),
                          num_units=self.word_dict_size, W=W_init, b=b_init, nonlinearity=softmax, name='recons_x')
        # (batch_size*sent_length, word_dict_size)


        # ------------------------------- Baseline ----------------------------------
        if theano.config.floatX == 'float32':
            self.b = theano.shared(np.float32(5.5))
        else:
            self.b = theano.shared(np.float64(5.5))

    def get_params(self, tag='save'):
        '''
        all the params are divided into two groups: 1) trainable 2) untrainable etc. batch normal layers
        when save the model, we need all the params, only trainable tag == None
        when calculate the gradients, only trainable tag == True
        '''
        e = get_all_params(self.l_ebd, trainable=True)
        c = get_all_params(self.l_y, trainable=True)
        i = get_all_params(self.l_z, trainable=True)
        g = get_all_params(self.l_rec, trainable=True)
        if tag == 'e':  # 'e' in paper, embedding
            return e
        elif tag == 'c':  # 'w' in paper, classifier parameters
            return c
        elif tag == 'i':  # '\phi' in paper, inference parameters
            return i
        elif tag == 'g': # '\theta' in paper, generation parameters
            return g
        elif tag == 'all':  # all parameters for gradient
            return e + c + i + g
        else:  # 'save' for save model
            return get_all_params([self.l_ebd, self.l_y, self.l_z, self.l_rec])

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
            raise ValueError("mismatch: got %d values to set %d parameters" %(len(values), len(params)))
        for p, v in zip(params, values):
            if p.get_value().shape != v.shape:
                raise ValueError("shape mismatch")
            else:
                p.set_value(v)

    def load_pretrain(self, pretrain_path):
        # the loaded data has certain format
        import cPickle as pkl
        print "Loading pretraining parameters from " + pretrain_path
        with open(pretrain_path, 'rb') as f:
            load_wemb = pkl.load(f)
            wemb = self.l_ebd.get_params()
            for i in range(len(load_wemb)):
                # print(load_wemb[i].shape)
                # print(wemb[i].get_value().shape)
                wemb[i].set_value(load_wemb[i])

            load_lstm = pkl.load(f)
            lstm = self.l_c.get_params()
            for i in range(len(load_lstm)):
                # print(lstm[i].get_value().shape)
                # print(load_lstm[i].shape)
                lstm[i].set_value(load_lstm[i])

    def _forward_sents(self, incomings, dev_stage=False):
        '''
        Get embeddings for each word.
        we shift the sentences in decoder all the time.
        when train stage && keep_rate < 1.0, implement word dropout
        when ( dev stage && not inputless model) || keep_rate == 1.0, no word dropout
        incomings: (batch_size * max_sent_length) index of words
        '''
        enc_sents = get_output(self.l_ebd, {self.l_sents_in: incomings}, deterministic=dev_stage)
        sents_shift = T.zeros_like(enc_sents)
        sents_shift = T.set_subtensor(sents_shift[:, 1:, :], enc_sents[:, :-1, :])

        if (dev_stage and self.keep_rate != 0.0) or self.keep_rate == 1.0:
            return enc_sents, sents_shift, T.sum(0)

        _srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        keep_position = _srng.binomial((sents_shift.shape[0], sents_shift.shape[1]), n=1,
                                       p=self.keep_rate, dtype=theano.config.floatX)
        dec_sents = sents_shift * keep_position[:, :, None]
        return enc_sents, dec_sents, T.sum(1.0 - keep_position).astype('int64')

    def _forward_classifier(self, incomings, label=None, dev_stage=False):
        '''
        Calculate classifier results: y_pred.
        For unlabeled data, we will sample later.
        For labeled data, we will compute the classification loss.
        incomings = [enc_sents, mask] (batch_size, max_sent_length, word_embd_dims), (batch_size, max_sent_length)
        '''
        print "Classifier."
        enc_sents, mask = incomings
        y_pred = get_output(self.l_y, {self.l_enc_sents_in: enc_sents, self.l_mask_in: mask},
                            deterministic=dev_stage)
        # calculate loss and accuracy
        if label is None:  # if None then unlabeled data, calculate the cross entropy.
            label = y_pred
        loss_classifier = categorical_crossentropy(y_pred, label)  # \sum{ y*log(prob_y) }
        acc = categorical_accuracy(y_pred, label)  # useless if unlabeled data
        acc = T.mean(acc.astype(theano.config.floatX))
        return y_pred, loss_classifier, acc

    def _forward_encoder(self, incomgins, dev_stage=False):
        '''
        Forward encoder network.
        incomings = [enc_sents, mask, label]
        For labeled data, label is given
        For unlabeled data, label is sampled form y_pred
        return inference results
        '''
        print "Eocnder."
        enc_sents, mask, label = incomgins
        z, z_mu, z_var = get_output([self.l_z, self.l_z_mu, self.l_z_var],
                                    {self.l_enc_sents_in: enc_sents, self.l_mask_in: mask,
                                     self.l_label_in: label}, deterministic=dev_stage)
        return z, z_mu, z_var

    def _forward_decoder(self, incomings, dev_stage=False):
        '''
        Forward decoder network
        incomings = [label, z, dec_sents, mask]
        For labeled data, label is given.
        For unlabeled data, label is sampled form y_pred.
        return the reconstruction results
        '''
        print "Decoder."
        label, z, dec_sents, mask = incomings
        rec = get_output(self.l_rec, {self.l_label_in: label, self.l_z_in: z,
                        self.l_dec_sents_in: dec_sents, self.l_mask_in: mask},
                         deterministic=dev_stage)
        return rec

    def _sample_one_category(self, y_pred):
        '''
        Randomly sample one category form y_pred (batch_size, dim_y).
        Return one-hot representation of the sample (batch_size, dim_y).
        Only used in unlabeled data case. (NOT labeled data or dev_stage)
        return:
        label_onehot (bs, dim_y) one hot representation for sampled yi.
        y_pred_sampled (bs, ) each element is the prob of sampled yi.
        !
        From a perspective of Reinforcement Learning, this is a on-policy learning.
        Actually we can use \epsilon-greedy to sample from p(y|x).
        '''
        print "Sample to estimate Expectation term."
        _srand = RandomStreams(seed=lasagne.random.get_rng().randint(1, 2147462579))
        label_onehot = _srand.multinomial(n=1, pvals=y_pred, dtype=theano.config.floatX)  #(bs, dim_y)
        '''
        # grads of 'argmax' ?
        label_pos = T.argmax(label_onehot, axis=1)
        label_pos += T.arange(label_onehot.shape[0], dtype='int64') * label_onehot.shape[1]
        y_pred_sampled = y_pred.flatten()[label_pos]
        '''
        # if one time sample, use 'sum' instead of 'argmax'
        y_pred_sampled = T.sum(y_pred * label_onehot, axis=1)
        return label_onehot, y_pred_sampled

    def _get_baselines(self, incomings):
        # linear baseline
        '''
        From a perspective of Reinforcement Learning, -L(x,y) is the 'Reward' for unlabeled data,
        which is usually in positive value.
        We add a huge baseline to make this part positive.
        '''
            #loss_rec -= 2000.0
        # kl loss
        x, emb, mask = incomings
        return self.b * T.ones([mask.shape[0]])

    def cost_label(self, incomings, dev_stage=False):
        '''
        Cost function for labeled data.
        L(x,y) = - E_{q(z|x,y; \phi,e)}[log(p(x|y,z; \theta,e)] + D_kl (q(z|x,y; \phi,e) || p0(z))

        1. for the first term, we use MC-1 and reparameterization tick:
            - E_{q(z|x,y; \phi,e)}[log(p(x|y,z; \theta,e)] ~=
            - log p(x|y,z_sampled; \theta,e) - Baseline(x; \lambda)

        2. for the second term, the KL divergence is directly calculated:
            for q(z|x,y; \phi,e) ~ N(z_mu, z_var)
                p0(z) ~ N(0,1)
            D_kl = -\int_z { q(z|x,y; \phi,e) * log p0(z)}
                  +\int_z { q(z|x,y; \phi,e) * log q(z|x,y; \phi,e)}

        incomings = [sents, enc_sents, dec_sents, mask, label]
        we get label after _forward_classifier.
        '''
        print "Cost function for labeled data."
        sents, enc_sents, dec_sents, mask, label = incomings
        z, z_mu, z_var = self._forward_encoder([enc_sents, mask, label], dev_stage=dev_stage)
        rec = self._forward_decoder([label, z, dec_sents, mask], dev_stage=dev_stage)
        # reconstruction loss
        loss_rec = categorical_crossentropy(rec, sents.flatten())  # (batch_size*max_sent_length)
        loss_rec = (loss_rec.reshape((-1, mask.shape[1])) * mask).sum(-1)
        # batch ppl
        batch_ppl = T.exp(T.sum(loss_rec) / T.sum(mask))

        loss_kl = - GaussianMarginalLogDensity([z_mu, z_var], normal_priori=True)\
                  + GaussianMarginalLogDensity([z_mu, z_var], normal_priori=False)

        return loss_rec, loss_kl, batch_ppl

        
    def cost_unlabel_expectation(self, incomings, dev_stage=False):
        '''
        cost function for labeled data
        U(x) = E_{q(y|x;w,e)}[ L(x,y;\phi,\theta,e)] - H(q(y|x));
        We only calculate the first expectation term
        incomings = [enc_sents_u, dec_sents_u, mask_u, y_pred_u]
        '''
        print "Cost function for unlabeled data, calculate the Expectation term."
        sents_u, enc_sents_u, dec_sents_u, mask_u, y_pred_u = incomings
        loss_rec_u, loss_kl_u = T.zeros((mask_u.shape[0],)), T.zeros((mask_u.shape[0],))
        for i in range(self.dim_y):
            yi = T.zeros(y_pred_u.shape, dtype=theano.config.floatX)
            yi = T.set_subtensor(yi[:, i], 1.0)
            loss_rec_yi, loss_kl_yi, _ = \
                    self.cost_label([sents_u, enc_sents_u, dec_sents_u, mask_u, yi], dev_stage=dev_stage)
            loss_rec_u = loss_rec_u / T.sum(mask_u, axis=1)
            loss_rec_u += loss_rec_yi * y_pred_u[:, i]
            loss_kl_u += loss_kl_yi * y_pred_u[:, i]
        # end for
        batch_ppl = T.exp(T.sum(loss_rec_u) / T.sum(mask_u))
        return loss_rec_u, loss_kl_u, batch_ppl

    def train_expectation_function(self):
        '''
        unlabeled data train with expection
        '''
        print "Train Function: Calculate the Expectation of unlabeled data."
        sym_klw = T.scalar('sym_klw', dtype=theano.config.floatX)  # symbolic scalar of warming up
        sym_sents = T.matrix('sym_s', dtype='int64')
        sym_mask = T.matrix('sym_mask', dtype=theano.config.floatX)  # one hot!
        sym_label = T.matrix('sym_label', dtype=theano.config.floatX)
        sym_sents_u = T.matrix('sym_s_u', dtype='int64')
        sym_mask_u = T.matrix('sym_mask_u', dtype=theano.config.floatX)
        num_l, num_u = sym_sents.shape[0].astype(theano.config.floatX), \
                       sym_sents_u.shape[0].astype(theano.config.floatX)
        num_all = num_l + num_u

        # forward the network and get cost values
        enc_sents, dec_sents, _ = self._forward_sents(sym_sents, dev_stage=False)
        enc_sents_u, dec_sents_u, _ = self._forward_sents(sym_sents_u, dev_stage=False)

        # classifier loss
        y_pred, loss_class, acc = self._forward_classifier([enc_sents, sym_mask], sym_label, dev_stage=False)
        y_pred_u, loss_entropy, _ = self._forward_classifier([enc_sents_u, sym_mask_u], None, dev_stage=False)

        # reconstruction and kl loss
        loss_rec, loss_kl, ppl = self.cost_label([sym_sents, enc_sents, dec_sents, sym_mask, sym_label], dev_stage=False)
        loss_rec_u, loss_kl_u, ppl_u = self.cost_unlabel_expectation([sym_sents_u, enc_sents_u, dec_sents_u,
                                                        sym_mask_u, y_pred_u], dev_stage=False)
        
        # use baseline
        if self.use_baseline:
            baselines_u = self._get_baselines([sym_sents_u, enc_sents_u, sym_mask_u])
            loss_rec_u -= baselines_u
        
        total_cost = T.sum(loss_rec) + T.sum(loss_rec_u) - T.sum(loss_entropy)
        total_cost += sym_klw * (T.sum(loss_kl) + T.sum(loss_kl_u))
        total_cost += self.alpha * T.sum(loss_class) * num_all / num_l
        total_cost /= num_all

        all_params = self.get_params(tag='all')
        all_grads = theano.grad(total_cost, all_params)
        all_grads = total_norm_constraint(all_grads, max_norm=self.max_norm)
        updates = adam(all_grads, all_params, self.lr)

        train_input = [sym_sents, sym_mask, sym_label, sym_sents_u, sym_mask_u, sym_klw]
        train_output = [total_cost,
                        T.mean(loss_rec), T.mean(loss_rec_u), T.mean(loss_kl), T.mean(loss_kl_u),
                        T.mean(loss_class), T.mean(loss_entropy), ppl, ppl_u, acc, self.b]
        train_f = theano.function(inputs=train_input, outputs=train_output, updates=updates, name='train_expectation')
        return train_f

    def train_sample_function(self, ew=1.0):
        '''
        unlabeled data train with sample
        '''
        print "Train Function: Estimate the Expectation of unlabeled data by Sample."
        sym_klw = T.scalar('sym_klw', dtype=theano.config.floatX)  # symbolic scalar of warming up
        sym_sents = T.matrix('sym_s', dtype='int64')
        sym_mask = T.matrix('sym_mask', dtype=theano.config.floatX)  # one hot!
        sym_label = T.matrix('sym_label', dtype=theano.config.floatX)
        sym_sents_u = T.matrix('sym_s_u', dtype='int64')
        sym_mask_u = T.matrix('sym_mask_u', dtype=theano.config.floatX)
        num_l, num_u = sym_sents.shape[0].astype(theano.config.floatX), \
                       sym_sents_u.shape[0].astype(theano.config.floatX)
        num_all = num_l + num_u

        # forward the network and get cost values
        enc_sents, dec_sents, _ = self._forward_sents(sym_sents, dev_stage=False)
        enc_sents_u, dec_sents_u, _ = self._forward_sents(sym_sents_u, dev_stage=False)

        # classifier loss
        y_pred, loss_class, acc = self._forward_classifier([enc_sents, sym_mask], sym_label, dev_stage=False)
        y_pred_u, loss_entropy, _ = self._forward_classifier([enc_sents_u, sym_mask_u], None, dev_stage=False)
        sampled_label, y_pred_sampled = self._sample_one_category(y_pred_u)

        # reconstruction and kl loss
        loss_rec, loss_kl, ppl = self.cost_label([sym_sents, enc_sents, dec_sents, sym_mask, sym_label], dev_stage=False)
        loss_rec_u, loss_kl_u, ppl_u = self.cost_label([sym_sents_u, enc_sents_u, dec_sents_u,
                                                        sym_mask_u, sampled_label],
                                                        dev_stage=False)

        # use baseline
        # length normalization for unlabel
        const_Lxy = loss_rec_u / T.sum(sym_mask_u, axis=1) + sym_klw*loss_kl_u
        if self.use_baseline:
            baselines_u = self._get_baselines([sym_sents_u, enc_sents_u, sym_mask_u])
            const_Lxy -= baselines_u

        # gradients, see supplementary files for detail
        all_params, params_e, params_w, params_phi, params_theta = self.get_params(tag='all'), \
        self.get_params(tag='e'), self.get_params(tag='c'), self.get_params(tag='i'), self.get_params(tag='g')
        total_cost_directly = -T.sum(loss_entropy) * ew + T.sum(loss_rec + sym_klw * loss_kl)
        total_cost_directly += self.alpha * T.sum(loss_class) * num_all / num_l
        total_cost_directly /= num_all
        all_grads = theano.grad(total_cost_directly, all_params)
        grad_e = theano.grad(T.sum(const_Lxy * T.log(y_pred_sampled) + loss_rec_u + sym_klw * loss_kl_u) / num_all,
                             params_e, consider_constant=[const_Lxy])
        grad_w = theano.grad(T.sum(const_Lxy * T.log(y_pred_sampled)) / num_all,
                             params_w, consider_constant=[const_Lxy])
        grad_ig = theano.grad(T.sum(loss_rec_u + sym_klw * loss_kl_u) / num_all,
                             params_phi + params_theta, consider_constant=[const_Lxy])
        # combine the grads
        grad_unlabel = grad_e + grad_w + grad_ig
        all_grads = [gi + gj for gi, gj in zip(all_grads, grad_unlabel)]
        total_cost = total_cost_directly + T.sum(const_Lxy)/num_all  # not used in gradients

        '''
        # old cost function in AVAE
        all_params = self.get_params(tag='all')
        total_cost = T.sum(loss_rec) + T.sum(loss_rec_u) - T.sum(loss_entropy)
        total_cost += sym_klw * (T.sum(loss_kl) + T.sum(loss_kl_u))
        total_cost += self.alpha * T.sum(loss_class) * num_all / num_l
        total_cost /= num_all
        all_grads = theano.grad(total_cost, all_params)
        '''

        all_grads = [T.clip(g, -self.grad_clipping, self.grad_clipping) for g in all_grads]
        all_grads = total_norm_constraint(all_grads, max_norm=self.max_norm)
        updates = adam(all_grads, all_params, self.lr)
        update_baseline = {self.b: 0.9 * self.b + 0.1 * T.mean(loss_rec_u)}
        updates.update(update_baseline)
        train_input = [sym_sents, sym_mask, sym_label, sym_sents_u, sym_mask_u, sym_klw]

        train_output= [total_cost,
                       T.mean(loss_rec), T.mean(loss_rec_u), T.mean(loss_kl), T.mean(loss_kl_u),
                       T.mean(loss_class), T.mean(loss_entropy), ppl, ppl_u, acc, self.b]
        train_f = theano.function(inputs=train_input, outputs=train_output, updates=updates, name='train_sample')
        return train_f

    def test_function(self):
        '''
        only for labeled data.
        '''
        print "Dev/test function."
        sym_sents = T.matrix('sym_s', dtype='int64')
        sym_mask = T.matrix('sym_mask', dtype=theano.config.floatX)  # one hot!
        sym_label = T.matrix('sym_label', dtype=theano.config.floatX)

        # forward the network and calc the loss.
        enc_sents, dec_sents, _ = self._forward_sents(sym_sents, dev_stage=True)
        y_pred, loss_class, acc = self._forward_classifier([enc_sents, sym_mask], sym_label, dev_stage=True)
        loss_rec, loss_kl, ppl = self.cost_label([sym_sents, enc_sents, dec_sents, sym_mask, sym_label],
                                                 dev_stage=True)
        total_cost = loss_rec + loss_kl + self.alpha * loss_class
        valid_words = T.sum(sym_mask).astype('int64')

        # build functions
        test_input = [sym_sents, sym_mask, sym_label]
        test_output= [T.mean(total_cost), T.mean(loss_rec), T.mean(loss_kl), T.mean(loss_class), ppl, acc, valid_words]
        test_f = theano.function(inputs=test_input, outputs=test_output, name='test_function')
        return test_f



