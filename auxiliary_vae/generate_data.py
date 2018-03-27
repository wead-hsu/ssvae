import sys
import numpy as np
from deep_model import *
from utils.helper_functions import *
import theano
import lasagne
import cPickle as pkl
theano.optimizer_including='cudnn'
theano.exception_verbosity='high'

def init_config( ):
    params = {}
    params['data'] = 'imdb'   #'imdb' 'mds', etc
    params['data_path'] = '../data/imdb.semi.5000'
    params['weight_load_path'] = '../results/20160327-1739/39.weights'
    params['save_dir'] = '../results/'
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

    wdict = data['wdict']
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

    return [wdict, s_l_train, y_l_train, s_l_dev, y_l_dev, s_l_test, y_l_test]


def show(sample, sample_score, idict):
    #for i in range(len(sample)):
    for i in range(5):
        print 'score:', sample_score[i]
        #print sample[i]
        for j in sample[i]:
            print idict[j],


print "================ Initializing ================"
params = init_config()
wdict, s_l_train, y_l_train, s_l_dev, y_l_dev, s_l_test, y_l_test = load_data(params)
print "================= Modeling ================"
model = DeepModel( enc_num_units=params['enc_num_units'],dec_num_units=params['dec_num_units'],
                   dim_z=params['dim_z'], word_ebd_dims=params['word_ebd_dims'], word_dict_size=params['word_dict_size'],
                dim_y=params['dim_y'], dim_a=params['dim_a'], drop_out=0.0, keep_rate=1.0,
                   cost_beta=1.0,sample_unlabel=False)
model.build_model()

print "================= Model Weights ================"
cnt = 0
all_params = model.get_params()
for p in all_params:
    cnt+=1
    print cnt, p.name, p.get_value().shape

model.load_model(params['weight_load_path'])


random_gen = False

if random_gen:
    z = np.random.normal(0, 1, (1,1, params['dim_z'])).astype(theano.config.floatX)
else:
    encoding_path = '../results/forward.5000'
    encodings = pkl.load(open(encoding_path, 'rb'))
    z = encodings['z'][1:2]
    print z.shape

y = np.asarray([[0, 0]], dtype=theano.config.floatX)
print('y:', y)
print('z:', z)

idict = dict([(v, k) for k, v in wdict.items()])
idict[0] = '<EOS>'
idict[1] = '<UNK>'
sample, sample_score= model.beam_search(y, z, 100)
show(sample, sample_score, idict)
exit()
