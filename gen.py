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
import glob
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score


def ecac_run(X, n_clusters, data, pop_size=20, max_gens=2000, p_crossover=0.95, p_mutation=0.98,
             runs=10, y=None, log_file=False, evolutionary_plot=False):
    tifont = {'fontname': 'Times New Roman', 'fontsize': 20, 'fontweight': 'bold'}
    axfont = {'fontname': 'Times New Roman', 'fontsize': 16}

    for run in range(runs):
        print('============= TEST {} ============='.format(run + 1))
        print('Clustering started using ECAC'.format(data))
        print('Dataset: {}, Clusters: {}, Instances: {}, Features: {}'.format(data, n_clusters, len(X), len(X[0])))
        print('Population size: {}, Generations: {}'.format(pop_size, max_gens))

        start = time.time()
        population = []
        fit_log = []
        X = StandardScaler().fit_transform(X)

        print('Generating initial population')
        for _ in range(pop_size):
            individual = {'partition': random_gen(n_clusters, X)}
            individual['fitness'] = fitness_value(X, individual['partition'], n_clusters)
            if individual not in population:
                population.append(individual)
        best = sorted(population, key=lambda k: k['fitness'], reverse=True)[0]

        print('Starting genetic process')
        for i in range(max_gens):
            print('Generation {}'.format(i + 1))
            selected = []
            for _ in range(pop_size):
                selected.append(binary_tournament(population))
            children = reproduce(selected, pop_size, p_crossover, p_mutation, n_clusters)
            for j in range(len(children)):
                children[j]['fitness'] = fitness_value(X, children[j]['partition'], n_clusters)
            children.sort(key=lambda l: l['fitness'], reverse=True)
            if children[0]['fitness'] >= best['fitness']:
                best = children[0]
            population = children
            fit_log.append((i + 1, best['fitness']))

            if evolutionary_plot:
                plt.figure(figsize=(12, 8), dpi=200)
                plt.title('ECAC - Generation {}'.format(i+1), **tifont)
                plt.xlabel('Proanthocyanidins', **axfont)
                plt.ylabel('Total Phenols', **axfont)
                colors = best['partition']
                plt.scatter(X[:, 9], X[:, 6], c=colors, edgecolor='k', cmap='YlGnBu')
                plt.tight_layout()
                if not os.path.exists('figures/{}/{}'.format(data, run+1)):
                    os.makedirs('figures/{}/{}'.format(data, run+1))
                plt.savefig('figures/{}/{}/scatter_{}.jpg'.format(data, run+1, i+1), format='jpg')
            if best['fitness'] == 1:
                break

        run_time = time.time() - start
        best['time'] = run_time
        print('Optimization finished in {:.2f}s with an objective of {:.4f}'.format(best['time'], best['fitness']))
        best['partition'] = np.array(best['partition'])

        d = dict()
        d['Dataset'] = data
        d['Algorithm'] = 'ecac'
        d['Clusters'] = n_clusters
        d['Instances'] = len(X)
        d['Features'] = len(X[0])
        d['Pop. size'] = pop_size
        d['Max. gens'] = max_gens
        d['No. objectives'] = 1
        d['Obj. 1 name'] = 'generalization'
        d['Objective 1'] = best['fitness']
        d['Obj. 2 name'] = np.nan
        d['Objective 2'] = np.nan
        d['Time'] = best['time']
        if y is None:
            d['Adjusted Rand Index'] = np.nan
        else:
            d['Adjusted Rand Index'] = adjusted_rand_score(y, best['partition'])
        for i in range(len(best['partition'])):
            d['X{}'.format(i + 1)] = '{}'.format(best['partition'][i])

        out = pd.DataFrame(d, index=[data])
        if not os.path.exists('ecac-out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens)):
            os.makedirs('ecac-out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens))
        out.to_csv('ecac-out/{}_{}_{}_{}/solution-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size, max_gens,
                                                                             data, n_clusters, pop_size, max_gens,
                                                                             run + 1), index=False)
        if log_file:
            log = pd.DataFrame(fit_log, columns=['gen', 'fitness'])
            log.to_csv('ecac-out/{}_{}_{}_{}/log-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size, max_gens,
                                                                            data, n_clusters, pop_size, max_gens,
                                                                            run + 1), index=False)

        filenames = glob.glob("ecac-out/{}_{}_{}_{}/solution*".format(data, n_clusters, pop_size, max_gens))
        df = pd.DataFrame()
        for name in filenames:
            temp_df = pd.read_csv(name)
            df = df.append(temp_df)
        df.reset_index(drop=True, inplace=True)
        df.to_csv('ecac-out/solutions-{}_{}_{}_{}-{}.csv'.format(data, n_clusters,
                                                                 pop_size, max_gens, runs))
