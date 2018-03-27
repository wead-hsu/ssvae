import numpy as np
import cPickle
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

data_path = '../../results_backup/before-now/10000_2/forward.10000'

data =cPickle.load(open(data_path,'rb'))
X = data['z']
Y = data['y'][:,1].astype(np.int32)
target_names = ['POS','NEG']

pca = PCA(n_components=2)
X_p = pca.fit(X).transform(X)

tsne = TSNE(n_components=2)
X_t = tsne.fit_transform(X)

plt.figure()
for c, i, target_name in zip("rb", [0, 1], target_names):
    plt.scatter(X_p[Y == i, 0], X_p[Y == i, 1], c=c, label=target_name,s=10)
plt.legend()
plt.title('PCA')

plt.figure()
for c, i, target_name in zip("rb", [0, 1], target_names):
    plt.scatter(X_t[Y == i, 0], X_t[Y == i, 1], c=c, label=target_name,s=8)
plt.legend()
plt.title('TSNE for a')

plt.show()
