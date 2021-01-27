# ECAC
Source code of the "Evolutionary Clustering Algorithm using Classifiers" (ECAC), an evolutionary approach to clustering that takes advantage of supervised learning techniques. ECAC generates its initial population using both k-means and random individuals. A one-point crossover and a neighbor-biased mutation operators enhance the search for solutions while looking to maximize the algorithm's objective funciton. This function is constructed by three classifiers that take as training labels the assigned groups in an individual's chromosome, and the value returned by it is the average Area under the Curve achieved by the classifiers.

ECAC is avaible in this repository in a Python implementation.

# Algorithm parameters
``data``: a string with the name of the dataset used for naming the output file.  
``X``: array containing the dataset features with no header. Each row must belong to one individual with one column per feature.  
``n_clusters``: int with the number of desider clusters.  
``max_gens`` (default = 100): maximum generations in the evolutionary process.  
``pop_size`` (default = 100): population size that is carried along the evolutionary process.  
``p_crossover`` (default = 0.95): probability of running the crossover operator.  
``p_mutation`` (default = 0.98): probability of running the mutation operator.  

### Data retrieval function
An additional data retrieval function is included for easy access and generation of the parameters X, clusters, and data. The function will use the datasets included in the path ``/data`` and returns the data string, the X features and the dataset's number of reference classes (n_clusters). To run it on Python and get the information of the *wine* dataset, run these commands in the interface.  
``>>> from retr import *``  
``>>> data, n_clusters, X, _ = data_retrieval('wine')``  

The provided datasets in the ``/data`` path (therefore the options to run the data_retrieval function) are: breast-tissue, ecoli, forest, glass, iris, knowledge, segment, spambase, transfusion, and wine. Label files are included for every dataset for any desired benchmarking tests.

# Setup and run using Python
Open your prefered python interface and follow this commands to generate a clustering using ECAC. To execute it, just import the functions in gen.py and run ecac_run() with all of its parameters. See the example code below, which follows the data, n_clusters and X variables set previously for the *wine* dataset.  
**Important**: You will need to have previously installed some basic data science packages such as numpy, pandas and scikit-learn.

``>>> from gen import *``  
``>>> solution = ecac_run(data, X, n_clusters, max_gens=100, pop_size=100, p_crossover=0.95, p_mutation=0.98)``  

Running these commands will execute ECAC using the wine dataset's features, 3 clusters, 100 generations, 100 individuals per population and probabilities of running the crossover and mutation operators of 0.95 and 0.98. A dictionary is returned containing an array with the partition, the solution's fitness and the run time for getting the solution and a .csv file is stores in the ``/ecac-out`` path with the test information and output.
