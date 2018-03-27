import numpy as np
import cPickle

#data_path = '../../results_backup/before-now/20000_1/forward.20000'
data_path = '../../results_backup/before-now/10000_adgm/forward.10000'

data =cPickle.load(open(data_path,'rb'))

a = data['a']
u = data['u']
kl_p0z = data['kl_p0z']
kl_qz = data['kl_qz']
kl_qa = data['kl_qa']
kl_pa = data['kl_pa']

for i in np.sort(kl_qz-kl_p0z):
    print i
print '.'
print np.sort(kl_qa-kl_pa)

