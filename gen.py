"""
@authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, Francisco Javier Cantu-Ortiz
"""

from ind import *
from obj import *
from oper import *
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import time
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
def ecac_run(data, X, n_clusters, max_gens=100, pop_size=100, p_crossover=0.95, p_mutation=0.98):
    print('Dataset: {}, Clusters: {}, Instances: {}, Features: {}'.format(data, n_clusters, len(X), len(X[0])))
    sta = time.time()
    X = StandardScaler().fit_transform(X)
    population = []
    print('Generating initial population')
    for _ in range(pop_size):
        individual = {'partition' : random_gen(n_clusters, X)}     
        individual['fitness'] = fitness_value(X, individual['partition'], n_clusters)
        if individual not in population: population.append(individual)
    best = sorted(population, key=lambda i: i['fitness'], reverse=True)[0]
    print('Genetic process running...')
    for i in range(max_gens):
        selected = []
        for _ in range(pop_size): selected.append(binary_tournament(population))
        children = reproduce(selected, pop_size, p_crossover, p_mutation, n_clusters)
        for j in range(len(children)):
            children[j]['fitness'] = fitness_value(X, children[j]['partition'], n_clusters)
        children.sort(key=lambda i: i['fitness'], reverse=True)
        if children[0]['fitness'] >= best['fitness']: best = children[0]
        population = children
        if best['fitness'] == 1: break
    fin = time.time()
    tim_dif = fin-sta
    best['time'] = tim_dif
    print('Optimization finished in {:.2f}s with an objective of {:.4f}'.format(best['time'], best['fitness']))
    best['partition'] = np.array(best['partition'])
    d = {}
    d['Clusters'] = n_clusters
    d['Population size'] = pop_size
    d['Max. gens'] = max_gens
    d['Objective'] = '{:.6}'.format(best['fitness'])
    d['Time'] = '{:.6}'.format(best['time'])
    for i in range(len(best['partition'])):
        d['X{}'.format(i+1)] = '{}'.format(best['partition'][i])
    out = pd.DataFrame(d, index=[data])
    n_3d = '{0:03}'.format(n_clusters)
    gens_3d = '{0:03}'.format(max_gens)
    pops_3d = '{0:03}'.format(pop_size)
    if not os.path.exists('ecac-out'): os.makedirs('ecac-out')
    out.to_csv('ecac-out/solution-{}_{}_{}_{}.csv'.format(data, n_3d, gens_3d, pops_3d))
    return best
