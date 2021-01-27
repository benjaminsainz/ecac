"""
@authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, Francisco Javier Cantu-Ortiz
"""

from sklearn.cluster import KMeans
import numpy as np
def km_gen(n_clusters, X):
    kmeans = KMeans(n_clusters, max_iter=np.random.randint(1,100)) # alternative: init='random'
    KMmodel = kmeans.fit(X)
    return list(KMmodel.labels_)

def random_gen(n_clusters, X):
    k_set = []
    for i in range(n_clusters): k_set.append(i)
    ind=[]
    for i in range(len(X)): ind.append(k_set[np.random.randint(0,len(k_set))])
    return ind
