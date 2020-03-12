from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

input_file = "Homework 1/train.csv"

# comma delimited is the default
df = pd.read_csv(input_file, header = 0)
# put the original column names in a python list
original_headers = list(df.columns.values)
# remove the non-numeric columns
df = df._get_numeric_data()
# put the numeric column names in a python list
numeric_headers = list(df.columns.values)
# create a numpy array with the numeric values for input into scikit-learn
numpy_array = df.as_matrix()

x = numpy_array[:, 2:]
y = numpy_array[:,1:]

print(x)

clf = Ridge(alpha=1.0)
clf.fit(x, y)
