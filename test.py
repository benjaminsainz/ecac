"""
@authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, Francisco Javier Cantu-Ortiz
"""

from retr import *
from gen import *


def test(ds, pop_size=20, max_gens=2000, runs=10):
    for d in ds:
        data, n_clusters, X, y = retrieval(d)
        ecac_run(X, n_clusters, data, pop_size, max_gens, runs=runs, y=y, log_file=False, evolutionary_plot=False)


paper = ['breast-tissue', 'dermatology', 'ecoli', 'forest', 'glass', 'iris', 'leaf', 'liver', 'transfusion', 'wine']

test(paper)
