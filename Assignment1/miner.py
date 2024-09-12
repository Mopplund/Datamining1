import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('iris_data.csv', sep=';')
labels = pd.read_csv('iris_labels.csv', sep=';')

print(data.shape)
print(data.head())
print('===========')

data = pd.merge(data, labels, on = 'id', how = 'inner')

print(data.head())
print('===========')

data.drop(['examiner'], axis = 1, inplace = True)

print(data.head())
print('===========')

data = data.sort_values('species')

print(data.head())
print('===========')


sns.pairplot(data, hue = "species")
plt.show()