# Data Preprocessing Template

# Importing the libraries
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../datasets/Position_Salaries.csv')
# Features
X = dataset.iloc[:, 1:2].values
# Dependent Variable, the thing that we want to predict
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# 1/3 of the dataset will be used to test, the rest will be trained
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
# Most alg has the feature scaling inside, however SVR class does not have the function build-in
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print(y_pred)

# visualize the SVR
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.show()




