import sys
import scipy.io
import numpy as np
from deep_model import SentGenLayer
from helper_functions import *
import theano
import lasagne
import time
from datetime import datetime
theano.optimizer_including='cudnn'
theano.exception_verbosity='high'

def init_config( ):
    params = {}
    params['model'] = 'Train'   #'Generate'
    params['data_path'] = '../data/imdb_usey.dict20000.lstm100.sents2'
    params['exp_time'] = str(datetime.now().strftime('%Y%m%d-%H%M'))
    params['save_dir'] = '../results/'+params['exp_time']
    params['opt_function'] = 'adam' #adam adagrad adadelta
    params['lr'] = 0.0004
    params['beta1'] = 1e-1
    params['beta2'] = 1e-3
    params['decay_rate'] = 1e-6
    params['num_epoches'] = 42
    params['annealing_center'] = 38.0
    params['annealing_width'] = 3.0 #*10
    params['keep_rate'] = 0.70 #>1 keep all <0 inputless model
    params['drop_out'] = 0.2
    params['dev_per'] = 1
    params['save_per'] = 3
    params['batch_size'] = 40
    params['dim_z'] = 50
    params['use_y'] = True
    params['dim_y'] = 0
    params['enc_num_units'] = 512
    params['dec_num_units'] = 512
    params['use_peepholes'] = False
    params['use_fixed_ebd'] = False #True using glove; False using trained ebd
    params['ebd_dim'] = 256

    params['is_load_weght'] = False
    params['weight_load_path'] = '../../1'

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
    y_train = data['y_train'].astype( theano.config.floatX )
    y_dev = data['y_dev'].astype( theano.config.floatX )
    params['dim_y'] = y_train.shape[1]

    return [w_dict, w_idict, word_ebd, s_train, mask_train, y_train, s_dev, mask_dev, y_dev]

print "================ Initializing ================"
params = init_config()
w_dict, w_idict, word_ebd, s_train, mask_train, y_train, s_dev, mask_dev, y_dev = load_data(params)
print params
print theano.config.optimizer

print "================= Modeling ================"
l_inp_sents = lasagne.layers.InputLayer((params['max_sen_length'],params['batch_size']))
l_vae = SentGenLayer( input_layer_sentences=l_inp_sents,
                      batch_size=params['batch_size'], dimz=params['dim_z'], max_sen_length=params['max_sen_length'],
                      enc_num_units=params['enc_num_units'], dec_num_units=params['enc_num_units'],
                      word_ebds=word_ebd, num_all_words=params['num_all_words'], ebd_dim=params['ebd_dim'],
                      use_y=params['use_y'], dimy=params['dim_y'],
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

print "================= Batching =================="
n_batches_train = params['num_train']/params['batch_size']
n_batches_test = params['num_dev']/ params['batch_size']
batchitertrain = BatchIterator(range(params['num_train']), params['batch_size'], data=(s_train, mask_train, y_train))
batchitertrain = threaded_generator(batchitertrain,3)
batchitertest = BatchIterator(range(params['num_dev']), params['batch_size'], data=(s_dev, mask_dev, y_dev))
batchitertest = threaded_generator(batchitertest,3)

print "================= Compiling ====================== "
sym_s = T.imatrix('sym_s')
sym_mask = T.matrix('sym_mask')
sym_w = T.fscalar('sym_w')
sym_y = T.matrix('sym_y')

sym_s.tag.test_value = np.zeros((params['max_sen_length'],params['batch_size']),dtype=np.int32)
sym_mask.tag.test_value = np.zeros((params['max_sen_length'],params['batch_size']), dtype=theano.config.floatX)
sym_y.tag.test_value = np.zeros((params['batch_size'],params['dim_y']), dtype=theano.config.floatX)
sh_s = theano.shared(np.zeros((params['max_sen_length'],params['batch_size']),dtype=np.int32))
sh_mask = theano.shared(np.zeros((params['max_sen_length'],params['batch_size']), dtype=theano.config.floatX))
sh_w = theano.shared( np.float32(0.0) )
sh_y = theano.shared(np.zeros((params['batch_size'],params['dim_y']), dtype=theano.config.floatX))

print "================= Compiling Theano.functions ====================== "
givens = [(sym_s, sh_s),(sym_mask, sh_mask),(sym_w, sh_w),(sym_y, sh_y)]
import theano.gradient
s_clip = theano.gradient.grad_clip(sym_s, -10.0, 10.0) # see graves generating sequences
cost = l_vae.get_cost(params['keep_rate'], params['drop_out'], s_clip, sym_mask, sym_w, sym_y)
all_grads = theano.grad(cost,all_params)
all_grads, step_norm, multiplier = step_clipping(all_grads, threshold=10.0, to_zero=False)
if params['opt_function'] == 'adam':
    updates, steps = adam(all_grads, all_params, learning_rate=params['lr'], beta1=params['beta1'],\
                          beta2=params['beta2'],decay_factor=1.0-params['decay_rate'] )
elif params['opt_function'] == 'adagrad':
    updates, steps = adagrad(all_grads, all_params, learning_rate=params['lr'])
else:
    updates, steps = adadelta(all_grads, all_params)

outputs =  [cost, step_norm] + l_vae.get_train_results()
train = theano.function([],outputs+ [multiplier], givens=givens, updates=updates)
#eval outputs
if params['keep_rate'] > 0:# <=0 means inputless
    eval_result = l_vae.get_cost(2.0, -1.0, sym_s, sym_mask, sym_w, sym_y)
else:
    eval_result = l_vae.get_cost(-2.0, -1.0, sym_s, sym_mask, sym_w, sym_y)

eval_outputs = [eval_result] + l_vae.get_train_results()
eval_cost = theano.function([],eval_outputs, givens=givens)

print "================= Training ====================== "
if not os.path.exists(params['save_dir']):
    os.makedirs(params['save_dir'])
log_file_path = params['save_dir'] +os.path.sep + params['exp_time'] + '-details.log'
res_file_path = params['save_dir'] +os.path.sep + 'train-results.log'
log_write_line(log_file_path,str(params),"a")
log_write_line(res_file_path,str(params),"a")

for epoch in range( params['num_epoches']):
    start_time = time.time()
    tol_cost, tol_kl, tol_recons, tol_w_num = 0.0, 0.0, 0.0, 0.0
    for i in range(n_batches_train):
        s,m,y = batchitertrain.next()
        sh_s.set_value( s.transpose(), borrow=True )
        sh_mask.set_value( m.transpose(), borrow=True)
        sh_y.set_value(y, borrow=True)
        #
        anneal_value = np.float32(epoch)+ np.float32(i)/np.float32(n_batches_train) - params['annealing_center']
        anneal_value /= params['annealing_width']
        if anneal_value >= 7.0:
            kl_w = 1.0
        else:
            kl_w = np.exp(anneal_value)/( 1.0 + np.exp(anneal_value))
        sh_w.set_value(np.float32(kl_w), borrow=True)

        cc, total_norm, loss_kl, loss_recons, ppl, w_num, drop_num, multi = train()
        tol_cost,tol_kl,tol_recons,tol_w_num = tol_cost + cc,tol_kl + loss_kl,tol_recons + loss_recons,tol_w_num + w_num
        out_str = "Epoch %d TrainBatch %d: cost: %f kl_loss: %e recons_loss: %f ppl: %f total L2: %f multi: %f kl_w: %f dec_dropout: %d" \
                  %(epoch, i, cc, loss_kl, loss_recons, ppl, total_norm, multi, kl_w, drop_num)
        print out_str
        log_write_line( log_file_path,out_str, "a")
    out_str = "TRAIN Epoch %d: avg_cost: %f avg_kl: %f avg_recons: %f avg_ppl: %f time: %f " \
              %(epoch, tol_cost/n_batches_train, tol_kl/n_batches_train, tol_recons/n_batches_train,\
                np.exp(tol_recons*np.float(params['batch_size'])/tol_w_num), time.time() - start_time)
    print out_str
    log_write_line( log_file_path,out_str, "a")
    log_write_line( res_file_path,out_str,"a")

    start_time = time.time()
    tol_cost, tol_kl, tol_recons, tol_w_num = 0.0, 0.0, 0.0, 0.0
    if (epoch + 1) % params['dev_per'] == 0:
        for i in range(n_batches_test):
            s,m,y = batchitertest.next()
            sh_s.set_value( s.transpose(), borrow=True )
            sh_mask.set_value( m.transpose(), borrow=True)
            sh_w.set_value( np.float32(1.0),borrow=True)
            sh_y.set_value(y, borrow=True)

            cc, loss_kl, loss_recons, ppl, w_num, drop_num = eval_cost()
            tol_cost,tol_kl,tol_recons,tol_w_num = tol_cost + cc,tol_kl + loss_kl,tol_recons + loss_recons,tol_w_num + w_num
            out_str = "Epoch %d TestBatch %d: cost: %f kl_loss: %e recons_loss: %f ppl: %f dec_dropout: %d " \
                      %(epoch, i, cc, loss_kl, loss_recons, ppl, drop_num)
            print out_str
            log_write_line( log_file_path,out_str, "a")
        out_str = "DEV>>>Epoch %d: avg_cost: %f avg_kl: %f avg_recons: %f avg_ppl: %f time: %f <<<DEV"\
                  %(epoch, tol_cost/n_batches_test, tol_kl/n_batches_test, tol_recons/n_batches_test,\
                    np.exp(tol_recons*np.float(params['batch_size'])/tol_w_num), time.time() - start_time)
        print out_str
        log_write_line( log_file_path,out_str, "a")
        log_write_line( res_file_path,out_str,"a")

    if (epoch +1 )% params['save_per'] == 0:
        save_params(l_vae, params['save_dir'] + os.path.sep + str(epoch)+".weights" )







