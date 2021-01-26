# ECAC
Source code of the "Evolutionary Clustering Algorithm using Classifiers" (ECAC), an evolutionary approach to clustering that takes advantage of supervised learning techniques. ECAC generates its initial population using both k-means and random individuals. A one-point crossover and a neighbor-biased mutation operators enhance the search for solutions while looking to maximize the algorithm's objective funciton. This function is constructed by three classifiers that take as training labels the assigned groups in an individual's chromosome, and the value returned by it is the average Area under the Curve AUC obtained by the classifiers.

ECAC is avaible in this repository in a Python implementation.
