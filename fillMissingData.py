# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # : takes all of the columns - 1
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3]) # Upper bound is excluded, so we put 3 to take indexes 1 and 2, starting at 0
X[:, 1:3] = imputer.transform(X[:, 1:3])
