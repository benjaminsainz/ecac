# ECAC
**Authors:** Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez, Hector G. Ceballos, Francisco J. Cantu-Ortiz  
**Paper title:** Evolutionary Clustering Algorithm Using Supervised Classifiers  

Source code of the Evolutionary Clustering Algorithm using Classifiers (ECAC) [1], an evolutionary approach to clustering that takes advantage of supervised learning techniques. ECAC generates its initial population using random individuals and a one-point crossover and neighbor-biased mutation operators enhance the search for solutions while looking to maximize the algorithm's objective function. This function is constructed by three classifiers that take as training labels the assigned groups in an individual's chromosome, and the value returned by it is the average Area Under the Curve achieved by the classifiers.

ECAC is available in this repository in a Python implementation.

# Algorithm hyper-parameters
``X``: an array containing the dataset features with no header. Each row must belong to one object with one column per feature.  
``n_clusters``: int with the number of desired clusters.  
``data``: a string with the name of the dataset used for printing the algorithm initialization and naming the output file.  
``pop_size`` (default = 20): population size that is carried along the evolutionary process.   
``max_gens`` (default = 2000): maximum generations in the evolutionary process.   
``p_crossover`` (default = 0.95): probability of running the crossover operator.  
``p_mutation`` (default = 0.98): probability of running the mutation operator.  
``runs`` (default = 10): independent runs of the algorithm.  
``y`` (default = None): one-dimensional array with the ground truth cluster labels if available.  
``log_file`` (default = False): creates a .csv file with the fitness value of the best individual per generation.  
``evolutionary_plot`` (default = False): creates multiple .jpg files with scatter plots of the first two columns from the dataset and their cluster membership.  

### Optional data retrieval function
An additional data retrieval function is included for easy access and generation of the parameters X, clusters and data. The function will use the datasets included in the path ``/data`` and returns the data string, the X features, and the dataset's number of reference classes (n_clusters). The only parameter for this function is a string with a dataset name from the options. To run it on Python and get the information of the *wine* dataset, run these commands in the interface.     
``>>> from retr import *``  
``>>> data, n_clusters, X, y = data_retrieval('wine')``  

The provided datasets in the ``/data`` path (therefore the options to run the data_retrieval function) are breast-tissue, ecoli, forest, glass, iris, knowledge, segment, spambase, transfusion, and wine. Label files are included for every dataset for any desired benchmarking tests.

# Setup and run using Python
Open your preferred Python interface and follow these commands to generate a clustering using ECAC. To execute it, just import the functions in *gen.py* and run ``ecac_run()`` with all of its parameters. See the example code below, which follows the data, n_clusters, X, and y variables set previously for the *wine* dataset.  
**Important**: You will need to have previously installed some basic data science packages such as numpy, pandas, and scikit-learn.

``>>> from gen import *``  
``>>> ecac_run(X, n_clusters, data, pop_size=20, max_gens=2000, p_crossover=0.95, p_mutation=0.98, runs=10, y=y, log_file=False, evolutionary_plot=False)``  

Running these commands will execute ECAC using the wine dataset's features, 3 clusters, 20 individuals per population, 2000 generations, probabilities of running the crossover and mutation operators of 0.95 and 0.98 for 10 independent runs, and will compute the adjusted RAND index between the solutions and the provided y array. No log files or evolutionary plots will be exported. A csv file is stored in the ``/ecac-out`` path with the test information and outputs.

A test.py file is provided for a more straight-forward approach to using the algorithm.  

I really hope ECAC is useful for your data mining tasks,  
Benjamin  
**LinkedIn:** https://www.linkedin.com/in/benjaminmariosainztinajero/   
**Email:** a01362640@itesm.mx, bm.sainz@gmail.com  

# References
[1] B. M. Sainz-Tinajero, A. E. Gutierrez-Rodriguez, H. G. Ceballos, and F. J. Cantu-Ortiz, “Evolutionary clustering algorithm using supervised classifiers,” in 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021. DOI: 10.1109/CEC45853.2021.9504826.
