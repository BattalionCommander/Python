import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
# Only needed if you want to display your plots inline if using Notebook
# change inline to auto if you have Spyder installed
#%matplotlib inline
# import some data to play with
iris = datasets.load_iris()
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

# Store the inputs as a Pandas Dataframe and set the column names
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']
