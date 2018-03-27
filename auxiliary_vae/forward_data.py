import sys
import numpy as np
from utils.helper_functions import *
import theano
import lasagne
theano.optimizer_including='cudnn'
theano.exception_verbosity='high'

def init_config( ):
    params = {}
    params['data'] = 'imdb'   #'imdb' 'mds', etc
    params['data_path'] = '../data/imdb.semi.2500'
    params['weight_load_path'] = '../data/imdb.semi.10k.ep37.w'
    params['model'] = 'SDGM' # 'ADGM' 'SDGM
    params['save_dir'] = '../results/'
    params['use_mean_lstm'] = False
    params['dim_z'] = 50
    params['dim_a'] = 20
    params['dim_y'] = 2
    params['batch_size'] = 100
    params['enc_num_units'] = 512
    params['dec_num_units'] = 512
    params['word_ebd_dims'] = 300

    return params

def load_data(params):
    data_path = params['data_path']
    import cPickle
    print "Loading data from " + data_path
    data =cPickle.load(open(data_path, "r"))

    s_l_train = data['s_l_train']
    y_l_train = data['y_l_train']
    s_l_dev = data['s_l_dev']
    y_l_dev = data['y_l_dev']
    s_l_test = data['s_l_test']
    y_l_test = data['y_l_test']

    params['word_dict_size'] = data['dict_size']
    params['num_train'] = len(s_l_train)
    params['num_dev'] = len(s_l_dev)
    params['num_test'] = len(s_l_test)

    return [s_l_train, y_l_train, s_l_dev, y_l_dev, s_l_test, y_l_test]

print "================ Initializing ================"
params = init_config()
if params['model'] == 'ADGM':
    from deep_model_adgm import *
else:
    from deep_model import *

s_l_train, y_l_train, s_l_dev, y_l_dev, s_l_test, y_l_test = load_data(params)
print "================= Modeling ================"
model = DeepModel( enc_num_units=params['enc_num_units'],dec_num_units=params['dec_num_units'],
                   dim_z=params['dim_z'], word_ebd_dims=params['word_ebd_dims'], word_dict_size=params['word_dict_size'],
                dim_y=params['dim_y'], dim_a=params['dim_a'], drop_out=0.0, keep_rate=1.0,
                   cost_beta=1.0,sample_unlabel=False)
model.build_model(use_mean_lstm=params['use_mean_lstm'], old_version=True)

print "================= Model Weights ================"
cnt = 0
all_params = model.get_params()
for p in all_params:
    cnt+=1
    print cnt, p.name, p.get_value().shape,
print 'END'

model.load_model(params['weight_load_path'])

print "================= Batching =================="
n_batches = int(params['num_train']/params['batch_size'])
bit = BatchIterator(params['num_train'], params['batch_size'], data=[s_l_train, y_l_train], testing=True)

print "================= Compiling ====================== "
print params
af = model.data_analysis_function()
ef = model.eval_function()
print "================= Forward data ====================== "
a, z, u, label, kl_p0z_by_unit, kl_qz_by_unit, kl_qa_by_unit, kl_pa_by_unit = [],[],[],[],[],[],[],[]
for i in range(n_batches):
    s,y = bit.next()
    s,m = make_mask(s,l_seq=800)
    cost, loss_recons, loss_kl, loss_classifier, batch_ppl, valid_words, acc = ef(s,m,y)
    a_i, _, _, z_i, _, _, u_i, _, _, kl_p0z_by_unit_i, kl_qz_by_unit_i, kl_qa_by_unit_i, kl_pa_by_unit_i = af(s,m,y)
    #a_i, _, _, z_i, _, _, kl_p0z_by_unit_i, kl_qz_by_unit_i, kl_qa_by_unit_i, kl_pa_by_unit_i = af(s,m,y)
    a.extend(a_i)
    z.extend(z_i)
    u.extend(u_i)
    label.extend(y)
    kl_p0z_by_unit.append(kl_p0z_by_unit_i)
    kl_qz_by_unit.append(kl_qz_by_unit_i)
    kl_qa_by_unit.append(kl_qa_by_unit_i)
    kl_pa_by_unit.append(kl_pa_by_unit_i)
    print acc
    print a_i.shape, z_i.shape, kl_p0z_by_unit_i.shape

print len(label)
print len(kl_qa_by_unit)

save_path = params['save_dir']+'forward.'+str(params['num_train'])
sd = {'a':np.asarray(a),'z':np.asarray(z),'u':np.asarray(u),'y':np.asarray(label),
      'kl_p0z':np.asarray(kl_p0z_by_unit).mean(0),'kl_qz':np.asarray(kl_qz_by_unit).mean(0),
      'kl_qa':np.asarray(kl_qa_by_unit).mean(0),'kl_pa':np.asarray(kl_pa_by_unit).mean(0)}
import cPickle
cPickle.dump(sd, open(save_path, "wb"))




