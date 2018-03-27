# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # Independent variable, num years
y = dataset.iloc[:, 1].values # Dependent variable, salary

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Simple Linear Regression: y = b0 + b1 * x1
# y = Dependent variable (DV) - trying to understand how this variable depends on something else
# x = Independent variable (IV) - assuming it causes the DV to change, or might not directly impact but implied association
# b1 = Coefficient for the IV - how the unit change in x will effect the unit change in y
# b0 = constant - point where line crosses vertical axis (no experience in this example)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# y_test is the real salaries of the 10 in the test set
# y_pred is the predicted salaries of the 10, predicted by the linear regression model
# Interesting to now compare the real vs. the predicted

# Visualizing the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # No need to change, we already trained on this line, would be the same if you changed it
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Our simple linear regression model does a good job at predicting new employees salaries
# Since there's not 100% linear dependency, we of course have a few outliers