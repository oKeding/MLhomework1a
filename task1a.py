import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error

input_file = "train.csv"
output_file = "submission.txt"

# comma delimited is the default
df = pd.read_csv(input_file, header = 0)
# put the original column names in a python list
original_headers = list(df.columns.values)
# remove the non-numeric columns
df = df._get_numeric_data()
# put the numeric column names in a python list
numeric_headers = list(df.columns.values)
# create a numpy array with the numeric values for input into scikit-learn
data = df.values

x = data[:, 2:]
y = data[:, 1:2]

n = len(data)
folds = 10
lambdas = [0.01, 0.1, 1, 10, 100]

mse = make_scorer(mean_squared_error)
solution = []

# Test with just splitting data for comparison:
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
clf = Ridge(1)
clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))

# Try with linear regression as well:
reg = LinearRegression().fit(X_train, y_train)
#print(reg.score(X_test, y_test))

# Do 10-fold cross validation and calculate average RMSE for each lambda:
clf = Ridge()
for l in lambdas:
    #print("lambda =", l, ":")
    clf.set_params(alpha=l)
    #R2s = cross_val_score(clf, x, y, cv=folds)
    #MSEs = cross_val_score(clf, x, y, cv=folds, scoring=mse)
    cv_results = cross_validate(clf, x, y, cv=folds, scoring=mse, return_estimator=True)
    
    #print("Coefficients:")
    #for model in cv_results['estimator']:
        #print(model.coef_)
    
    #print("Test MSEs:")
    MSEs = cv_results['test_score']
    
    RMSEs = np.sqrt(MSEs)
    #print("MSEs:", MSEs)
    #print("RMSEs:", RMSEs)
    
    #print("Average RMSE:", RMSEs.mean())
    print(RMSEs.mean())
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #print()
    
    solution.append(RMSEs.mean())

np.savetxt(output_file, solution, fmt='%.4f')