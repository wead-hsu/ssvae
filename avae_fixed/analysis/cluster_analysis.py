import numpy as np
import cPickle
import sklearn
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

text_path = '../../data/imdb.semi.2500'
data_path = '../../results_backup/before-now/10000_2/forward.2500'

def load_data(text_path):
    import cPickle
    print "Loading data from " + text_path
    data =cPickle.load(open(text_path, "r"))

    wdict = data['wdict']
    s_l_train = data['s_l_train']
    y_l_train = data['y_l_train']

    return [wdict, s_l_train, y_l_train]

wdict, s_l_train, y_l_train = load_data( text_path)
data =cPickle.load(open(data_path,'rb'))
X = data['z']
tsne = TSNE(n_components=3)
X = tsne.fit_transform(X)
cluster_num = 8
stat = np.zeros((cluster_num, 20000),dtype='float64')
cluster_label = SpectralClustering(n_clusters=cluster_num).fit(X).labels_
print len(cluster_label), len(s_l_train)

idict = dict()
for k in wdict.keys():
    idict[wdict[k]] = k
idict[0] = '<eos>'
idict[1] = '<unk>'

c_num = np.zeros((cluster_num,))
for i, s in enumerate(s_l_train):
    c_num[cluster_label[i]] += 1.0
    l = []
    for j in s:
       l.extend(j)

    l = set(l)
    for w in l:
        stat[cluster_label[i],w] += 1.0

tol_term_nums = np.sum(stat)

pc = c_num / np.sum(c_num)

ptc = stat / c_num[:,N
one]
pntc = pc[:,None] - ptc
#pt = np.sum(ptc,0,keepdims=True)
pt = np.sum(stat / stat.sum(),0,keepdims=True)
pnt = np.sum(pntc,0,keepdims=True)
ig = ptc * np.log((ptc/pc[:,None])/pt)
ig += pntc * np.log((pntc/pc[:,None])/pnt)
ig = np.sum(ig,axis=0)
sorted_ig = np.argsort(ig)[-1000:]
print sorted_ig
w_ig = [idict[i] for i in sorted_ig]
print w_ig

stat_ig = stat[:,sorted_ig]
#stat_ig = stat
pt_ig = np.sum(stat_ig/np.sum(stat_ig),0,keepdims=True)
ptc_ig = stat_ig / np.sum(stat_ig)
mi_ig = ptc_ig * np.log((ptc_ig/pt_ig)/pc[:,None])

sorted_miig =np.argsort(mi_ig)

for i in range(cluster_num):
    for j in range(40):
        print idict[sorted_miig[i,-(j+1)]],
    print ' END'








