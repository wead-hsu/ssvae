import sys
import numpy as np
from utils.helper_functions import *
import theano
import lasagne
import time
from datetime import datetime
theano.optimizer_including='cudnn'
theano.exception_verbosity='high'

def init_config( ):
    params = {}
    params['data'] = 'imdb'   #'imdb' 'mds', etc
    params['model'] = 'SDGM' # 'ADGM' 'SDGM-S' 'SDGM'
    params['data_path'] = '../data/imdb.semi.20000'
    params['webd_path'] = '../data/imdb.20k.glove.300'
    params['exp_time'] = str(datetime.now().strftime('%Y%m%d-%H%M'))+'-FIXED'
    params['save_dir'] = '../results/'+params['exp_time']
    params['lr'] = 0.0004
    params['num_epoches'] = 60
    params['klw_begin_from'] = 2
    params['cw_begin_from'] = 0  # warm up for classifier
    params['use_mean_lstm'] = True
    params['unlabel_stable'] = False #dev_stage=False -- use dropout in unlabel data
    params['old_version'] = True
    params['annealing_center'] = 50.0
    params['annealing_width'] = 8.0 #*7
    params['drop_out'] = 0.25
    params['beta'] = 1.0
    params['keep_rate'] = 0.90 #>1 keep all <0 inputless model
    params['dev_per'] = 1
    params['save_per'] = 2
    params['num_batches_train'] = 1250
    params['dev_batch_size'] = 50
    params['dim_z'] = 50
    params['dim_a'] = 20
    params['dim_y'] = 2
    params['enc_num_units'] = 512
    params['dec_num_units'] = 512
    params['word_ebd_dims'] = 300

    params['use_glove'] = True
    params['sample_unlabel'] = False
    params['is_load_weght'] = False
    params['weight_load_path'] = '../../1'
    #params['pretrain_load_path'] = '../data/pretrain_lm2.pkl'
    params['pretrain_load_path'] = None
    return params

def load_data(params):
    data_path = params['data_path']
    import cPickle
    print "Loading data from " + data_path
    data =cPickle.load(open(data_path, "r"))

    wdict = data['wdict']
    s_l_train = data['s_l_train']
    y_l_train = data['y_l_train']
    s_l_dev = data['s_l_dev']
    y_l_dev = data['y_l_dev']
    s_l_test = data['s_l_test']
    y_l_test = data['y_l_test']
    s_u = data['s_u']

    params['word_dict_size'] = data['dict_size']
    params['num_train'] = len(s_l_train)
    params['num_dev'] = len(s_l_dev)
    params['num_test'] = len(s_l_test)
    params['num_unlabel'] = len(s_u)

    if params['use_glove']:
        webd_path = params['webd_path']
        webd =cPickle.load(open(webd_path, "r"))
        vec_norm = webd['vec_norm']
        params['word_ebd_dims'] = vec_norm.shape[1]
    else:
        vec_norm = []

    return [wdict, vec_norm, s_l_train, y_l_train, s_l_dev, y_l_dev, s_l_test, y_l_test, s_u]

print "================ Initializing ================"
params = init_config()
if params['model'] == 'ADGM':
    from deep_model_adgm import *
elif params['model'] == 'SDGM-S':
    from deep_model_s import *
else:
    from deep_model import *

wdict, vec_norm, s_l_train, y_l_train, s_l_dev, y_l_dev, s_l_test, y_l_test, s_u = load_data(params)
print "================= Modeling ================"
model = DeepModel( enc_num_units=params['enc_num_units'],dec_num_units=params['dec_num_units'],
                   dim_z=params['dim_z'], word_ebd_dims=params['word_ebd_dims'], word_dict_size=params['word_dict_size'],
                dim_y=params['dim_y'], dim_a=params['dim_a'], drop_out=params['drop_out'],keep_rate=params['keep_rate'],
                   cost_beta=params['beta'], lr=params['lr'], sample_unlabel=params['sample_unlabel'])
if params['use_glove']:
    print 'Using glove...'
    model.build_model(use_mean_lstm=params['use_mean_lstm'], old_version=params['old_version'],
                      word_ebd_init= vec_norm)
else:
    model.build_model(use_mean_lstm=params['use_mean_lstm'], old_version=params['old_version'])

print "================= Model Weights ================"
cnt = 0
param_str = ''
all_params = model.get_params()
for p in all_params:
    cnt+=1
    print cnt, p.name, p.get_value().shape,
    param_str += str(cnt) + '. '+ str(p.name) + ': ' + str(p.get_value().shape) + '\n'
print 'END'

if params['is_load_weght']:
    model.load_model(params['weight_load_path'])
if params['pretrain_load_path']:
    print('load_pretraining')
    model.load_pretrain(params['pretrain_load_path'])

print "================= Batching =================="
params['train_batch_size'] = int(params['num_train']/params['num_batches_train'])
params['unlabel_batch_size'] = int(params['num_unlabel']/params['num_batches_train'])
params['num_batches_dev'] = params['num_dev']/ params['dev_batch_size']
params['num_batches_test'] = params['num_test']/ params['dev_batch_size']
batchitertrain = BatchIterator(params['num_train'], params['train_batch_size'], data=[s_l_train, y_l_train])
batchiterunlabel = BatchIterator(params['num_unlabel'], params['unlabel_batch_size'], data=[s_u])
batchiterdev = BatchIterator(params['num_dev'], params['dev_batch_size'], data=[s_l_dev,y_l_dev],testing=True)
batchitertest = BatchIterator(params['num_test'], params['dev_batch_size'], data=[s_l_test,y_l_test],testing=True)
print "================= Compiling ====================== "
print params
tf = model.train_function(semi_supervised=True, unlabel_stable=params['unlabel_stable'])
ef = model.eval_function()
print "================= Training ====================== "
if not os.path.exists(params['save_dir']):
    os.makedirs(params['save_dir'])
log_file_path = params['save_dir'] +os.path.sep + params['exp_time'] + '-details.log'
res_file_path = params['save_dir'] +os.path.sep + 'train-results.log'
log_write_line(res_file_path, param_str ,"a")
log_write_line(log_file_path,str(params),"a")
log_write_line(res_file_path,str(params),"a")

for epoch in range( params['num_epoches']):
    start_time = time.time()
    tol_cost, tol_recons, tol_kl,tol_words, tol_acc = 0.0, 0.0, 0.0, 0, 0.0
    n_batches_train = params['num_batches_train']
    for i in range(n_batches_train):
        s,y = batchitertrain.next()
        s,m = make_mask(s,l_seq=800)
        s_u = batchiterunlabel.next()[0]
        s_u, m_u = make_mask(s_u, l_seq=400)
        anneal_value = np.float32(epoch)+ np.float32(i)/np.float32(n_batches_train) - params['annealing_center']
        anneal_value /= params['annealing_width']
        if anneal_value >= 7.0:
            kl_w = 1.0
        elif anneal_value <= -7.0 or epoch < params['klw_begin_from']:
            kl_w = 1e-4
        else:
            kl_w = np.exp(anneal_value)/( 1.0 + np.exp(anneal_value))

        if epoch < params['cw_begin_from']:
            cw = 1e-4
        else:
            cw = 1.00

        outs = tf(s,m,y,s_u,m_u,np.float32(kl_w),np.float32(cw))
        cost, loss_recons, loss_recons_u, loss_kl, loss_kl_u, alpha, loss_classifier,\
        loss_entropy_u,batch_ppl, batch_ppl_u, valid_words, valid_words_u, word_drop_num, acc = outs
        tol_cost,tol_acc = tol_cost+cost,tol_acc+acc
        out_str = "Train %d Batch %d:\nLOSS:%f KL:%f KL_U:%f R:%f R_U:%f CAT:%f\n EPY:%f ACC:%f PPL:%f PPL_U:%f kl_w:%f alpha:%f" \
                  %(epoch,i,cost,loss_kl,loss_kl_u,loss_recons,loss_recons_u,loss_classifier,
                  loss_entropy_u,tol_acc/float(i+1),batch_ppl,batch_ppl_u,kl_w,alpha)
        print out_str
        log_write_line( log_file_path,out_str, "a")
    out_str = "Train Epoch %d: LOSS:%f ACC:%f Time:%f " \
              %(epoch, tol_cost/n_batches_train, tol_acc /n_batches_train, time.time() - start_time)
    print out_str
    log_write_line( log_file_path,out_str, "a")
    log_write_line( res_file_path,out_str, "a")
    #dev & test
    if (epoch + 1) % params['dev_per'] == 0:
        start_time = time.time()
        tol_cost, tol_recons, tol_kl, tol_words, tol_acc = 0.0, 0.0, 0.0,0, 0.0
        n_batches_dev = params['num_batches_dev']
        for i in range(n_batches_dev):
            s,y = batchiterdev.next()
            s,m = make_mask(s,l_seq=800)
            cost, loss_recons, loss_kl, loss_classifier, batch_ppl, valid_words, acc = ef(s,m,y)
            tol_cost, tol_recons, tol_kl, tol_words, tol_acc = \
            tol_cost+cost, tol_recons+loss_recons, tol_kl+loss_kl,\
            tol_words+valid_words, tol_acc+acc
            out_str = ">Dev %d Batch %d: LOSS:%f KL:%f RECONS:%f ACC:%f PPL:%f" \
                  %(epoch, i, cost, loss_kl, loss_recons, tol_acc/float(i+1), batch_ppl)
            print out_str
            log_write_line( log_file_path,out_str, "a")
        out_str = ">Dev Epoch %d: LOSS:%f KL:%f RECONS:%f PPL:%f ACC:%f Time:%f " \
              %(epoch, tol_cost/n_batches_dev, tol_kl/n_batches_dev, tol_recons/n_batches_dev,
                np.exp(tol_recons*np.float(params['dev_batch_size'])/tol_words), tol_acc /n_batches_dev, time.time() - start_time)
        print out_str
        log_write_line( log_file_path,out_str, "a")
        log_write_line( res_file_path,out_str,"a")

        start_time = time.time()
        tol_cost, tol_recons, tol_kl, tol_words, tol_acc = 0.0, 0.0, 0.0,0, 0.0
        n_batches_test = params['num_batches_test']
        for i in range(n_batches_test):
            s,y = batchitertest.next()
            s,m = make_mask(s,l_seq=800)
            cost, loss_recons, loss_kl, loss_classifier, batch_ppl, valid_words, acc = ef(s,m,y)
            tol_cost, tol_recons, tol_kl, tol_words, tol_acc = \
            tol_cost+cost, tol_recons+loss_recons, tol_kl+loss_kl,\
            tol_words+valid_words, tol_acc+acc
            out_str = ">Test %d Batch %d: LOSS:%f KL:%f RECONS:%f ACC:%f PPL:%f" \
                  %(epoch, i, cost, loss_kl, loss_recons, tol_acc/float(i+1), batch_ppl)
            print out_str
            log_write_line( log_file_path,out_str, "a")
        out_str = ">Test Epoch %d: LOSS:%f KL:%f RECONS:%f PPL:%f ACC:%f Time:%f " \
              %(epoch, tol_cost/n_batches_test, tol_kl/n_batches_test, tol_recons/n_batches_test,
                np.exp(tol_recons*np.float(params['dev_batch_size'])/tol_words), tol_acc /n_batches_test, time.time() - start_time)
        print out_str
        log_write_line( log_file_path,out_str, "a")
        log_write_line( res_file_path,out_str,"a")

    if (epoch +1 )% params['save_per'] == 0:
        model.save_model(params['save_dir'] + os.path.sep + str(epoch)+".weights" )
