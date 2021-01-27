"""
@authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, Francisco Javier Cantu-Ortiz
"""

from sklearn import datasets
import csv
import numpy as np
def data_retrieval(data):
    if data=='iris': 
        n_clusters=3
        X, y = datasets.load_iris().data, datasets.load_iris().target
    elif data=='wine':
        n_clusters=3
        X, y = datasets.load_wine().data, datasets.load_wine().target
    elif data=='ecoli':
        n_clusters=8
        X = list(csv.reader(open('data/ecoli_X.csv', newline='')))
        X = np.array(X)
        y = list(csv.reader(open('data/ecoli_y.csv', newline='')))
        y = [item for sublist in y for item in sublist]
        X, y = X.astype(np.float), np.array(y)
    elif data=='glass':
        n_clusters=6
        X = list(csv.reader(open('data/glass_X.csv', newline='')))
        X = np.array(X)
        y = list(csv.reader(open('data/glass_y.csv', newline='')))
        y = [item for sublist in y for item in sublist]
        X, y = X.astype(np.float), np.array(y)
    elif data=='transfusion':
        n_clusters=2
        X = list(csv.reader(open('data/transfusion_X.csv', newline='')))
        X = np.array(X)
        y = list(csv.reader(open('data/transfusion_y.csv', newline='', encoding='utf-8-sig')))
        y = [item for sublist in y for item in sublist]
        y = np.array(y)
        X, y = X.astype(np.float), y.astype(np.int)
    elif data=='forest':
        n_clusters=4
        X = list(csv.reader(open('data/forest_X.csv', newline='', encoding='utf-8-sig')))
        X = np.array(X)
        y = list(csv.reader(open('data/forest_y.csv', newline='', encoding='utf-8-sig')))
        y = [item for sublist in y for item in sublist]
        X, y = X.astype(np.float), np.array(y)
    elif data=='spambase':
        n_clusters=2
        X = list(csv.reader(open('data/spambase_X.csv', newline='', encoding='utf-8-sig')))
        X = np.array(X)
        y = list(csv.reader(open('data/spambase_y.csv', newline='', encoding='utf-8-sig')))
        y = [item for sublist in y for item in sublist]
        y = np.array(y)
        X, y = X.astype(np.float), y.astype(np.int)
    elif data=='segment':
        n_clusters=7
        X = list(csv.reader(open('data/segment_X.csv', newline='', encoding='utf-8-sig')))
        X = np.array(X)
        y = list(csv.reader(open('data/segment_y.csv', newline='', encoding='utf-8-sig')))
        y = [item for sublist in y for item in sublist]
        y = np.array(y)
        X, y = X.astype(np.float), y
    elif data=='breast-tissue':
        n_clusters=6
        X = list(csv.reader(open('data/breast-tissue_X.csv', newline='', encoding='utf-8-sig')))
        X = np.array(X)
        y = list(csv.reader(open('data/breast-tissue_y.csv', newline='', encoding='utf-8-sig')))
        y = [item for sublist in y for item in sublist]
        y = np.array(y)
        X, y = X.astype(np.float), y
    elif data=='leaf':
        n_clusters=36
        X = list(csv.reader(open('data/leaf_X.csv', newline='', encoding='utf-8-sig')))
        X = np.array(X)
        y = list(csv.reader(open('data/leaf_y.csv', newline='', encoding='utf-8-sig')))
        y = [item for sublist in y for item in sublist]
        y = np.array(y)
        X, y = X.astype(np.float), y.astype(np.int)

    elif data=='knowledge':
        n_clusters=4
        X = list(csv.reader(open('data/knowledge_X.csv', newline='', encoding='utf-8-sig')))
        X = np.array(X)
        y = list(csv.reader(open('data/knowledge_y.csv', newline='', encoding='utf-8-sig')))
        y = [item for sublist in y for item in sublist]
        y = np.array(y) 
        X, y = X.astype(np.float), y
    return data, n_clusters, X, y
