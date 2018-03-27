import sys
import scipy.io
import numpy as np
from deep_model import SentGenLayer
from helper_functions import *
import theano
import lasagne
import time
from datetime import datetime
theano.exception_verbosity='high'

def init_config( ):
    params = {}
    params['model'] = 'Generator'   #'Generate'
    params['data_path'] = '../data/ptb.glove50b.ebd'
    params['exp_time'] = str(datetime.now().strftime('%Y%m%d-%H%M'))
    params['save_dir'] = '../results/'+params['exp_time']
    params['lr'] = 0.0001
    params['beta1'] = 1e-3
    params['beta2'] = 1e-4
    params['decay_rate'] = 1e-4
    params['num_epoches'] = 100
    params['annealing_center'] = 20.0
    params['annealing_width'] = 2.0 #*10
    params['keep_rate'] = 0.65 #>1 keep all <0 inputless model
    params['dev_per'] = 1
    params['save_per'] = 10
    params['batch_size'] = 50
    params['dim_z'] = 15
    params['enc_num_units'] = 200
    params['dec_num_units'] = 200
    params['use_peepholes'] = False
    params['use_fixed_ebd'] = False #True using glove; False using trained ebd
    params['ebd_dim'] = 300

    params['is_load_weght'] = True
    params['weight_load_path'] = '../data/beam_search.weights'

    return params

def load_data(params):
    data_path = params['data_path']
    import cPickle
    print "Loading parameters from " + data_path
    data =cPickle.load(open(data_path, "r"))
    w_dict = data['w_dict']
    w_idict = data['w_idict']
    word_ebd = data['word_ebd'].astype( theano.config.floatX )
    s_train = data['s_train'].astype( np.int32 )
    mask_train = data['mask_train'].astype( theano.config.floatX )
    s_dev = data['s_dev'].astype( np.int32 )
    mask_dev = data['mask_dev'].astype( theano.config.floatX )

    params['num_train'] = s_train.shape[0]
    params['max_sen_length'] = s_train.shape[1]
    params['num_dev'] = s_dev.shape[0]
    params['num_all_words'] = word_ebd.shape[0]
    if params['use_fixed_ebd']:
        params['ebd_dim'] = word_ebd.shape[1]

    params['num_train'] = s_train.shape[0]

    return [w_dict, w_idict, word_ebd, s_train, mask_train , s_dev, mask_dev ]

print "================ Initializing ================"
params = init_config()
w_dict, w_idict, word_ebd, s_train, mask_train , s_dev, mask_dev = load_data(params)
print params
print theano.config.optimizer

print "================= Modeling ================"
l_inp_sents = lasagne.layers.InputLayer((params['max_sen_length'],params['batch_size']))
l_vae = SentGenLayer( input_layer_sentences=l_inp_sents,
                      batch_size=params['batch_size'], dimz=params['dim_z'], max_sen_length=params['max_sen_length'],
                      enc_num_units=params['enc_num_units'], dec_num_units=params['enc_num_units'],
                      word_ebds=word_ebd, num_all_words=params['num_all_words'], ebd_dim=params['ebd_dim'],
                      use_fixed_ebd=params['use_fixed_ebd'], use_peepholes=params['use_peepholes'])

print "================= Model Weights ================"
all_params = lasagne.layers.get_all_params(l_vae)
print "Check: # of all params:", len(all_params)
params_count = 1
for p in all_params:
    print params_count, p.name, p.get_value().shape
    params_count += 1

if params['is_load_weght'] or params['model'] == 'Generate':
    print "Loading Weights from:" +params['weight_load_path']
    load_params(l_vae, params['weight_load_path'])

print "================= Compiling Theano.functions ====================== "
gen_init = l_vae.gen_z_to_lm()
gen_step = l_vae.gen_step()

print "================= Generating ====================== "
log_file_path = '../results/gen-output.log'
for epoch in range( params['num_epoches']):
    z = (np.random.randn(params['dim_z'],)*1.01).astype('float32')
    sample, scores = l_vae.generator(z,30,gen_init, gen_step)
    sents = []
    for s in sample:
        sent = []
        for w in s:
            sent.append(w_idict[w])
        sents.append(sent)
    #
    sentences = [ ' '.join(s).lstrip().rstrip() for s in sents]
    for i in range(len(sentences)):
        strline = sentences[i] +' '+ str(scores[i])
        print strline
        log_write_line(log_file_path, strline, 'a')
    log_write_line(log_file_path, '-------------------------------------------------', 'a')





