"""
Authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, and Francisco Javier Cantu-Ortiz.
Year: 2021.
https://github.com/benjaminsainz/ecac
"""

import numpy as np


def random_gen(n_clusters, X):
    k_set = []
    for i in range(n_clusters):
        k_set.append(i)
    flag = False
    while flag is False:
        ind = []
        for i in range(len(X)):
            ind.append(k_set[np.random.randint(0, len(k_set))])
        flag = True
        for k in k_set:
            if k not in ind:
                flag = False
    return ind
