# ECAC
Source code of the "Evolutionary Clustering Algorithm using Classifiers" (ECAC), an evolutionary approach to clustering that takes advantage of supervised learning techniques. ECAC generates its initial population using both k-means and random individuals. A one-point crossover and a neighbor-biased mutation operators enhance the search for solutions while looking to maximize the algorithm's objective funciton. This function is constructed by three classifiers that take as training labels the assigned groups in an individual's chromosome, and the value returned by it is the average Area under the Curve AUC achieved by the classifiers.

ECAC is avaible in this repository in a Python implementation.

# Installation and run using Python
Open your prefered python interface and follow this commands to generate a clustering using ECAC. To execute it, just import the functions in gen.py and run ecac_run() with all of its parameters. See the example code below.
**Important**: You will need to have previously installed some basic data science packages such as numpy, pandas and scikit-learn.

``>>> from gen import *``  
``>>> from retr import *``  
data, n_clusters, X, _ = data_retrieval('iris') # dataset 
solution = ecac_run(data, X, n_clusters, max_gens=100, pop_size=100, p_crossover=0.95, p_mutation=0.98)


This script executes HG-means clustering in the Iris dataset, with 10 solutions in population, a maximum of 5000 iterations, 1 repetition, and 2, 5 and 10 clusters. Here the number of clusters is passed in an array, so values are separated by commas.
    
