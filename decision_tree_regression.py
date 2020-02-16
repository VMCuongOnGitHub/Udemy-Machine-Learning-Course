# Data Preprocessing Template

# Importing the libr
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datasets/Position_Salaries.csv')
# Features
X = dataset.iloc[:, 1:2].values
# Dependent Variable, the thing that we want to predict
y = dataset.iloc[:, 2].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# labelencoder_X = LabelEncoder()
# X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) #index of the category collumn
# ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# X = np.array(ct.fit_transform(X), dtype=np.int)
#
# # avoiding the dummy var trap
# X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# 1/3 of the dataset will be used to test, the rest will be trained
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# y_pred = regressor.predict(6.5)

# Visualize the Linear Regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.show()