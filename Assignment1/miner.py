import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA



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

print(f"Mean: {data['sl'].mean()}")
print(f"Std: {np.std(data['sl'])}")

print('===========')

print(data.value_counts("species"))

print('===========')


data = data[data["sl"] != -9999]
data = data[data["sw"] != -9999]

print(f"sl mean: {data['sl'].mean()}")
print(f"sl std: {np.std(data['sl'])}")

print(f"sw mean: {data['sw'].mean()}")
print(f"sw std: {np.std(data['sw'])}")

# Calculate the Z-scores of 'sl'
sl_z = stats.zscore(np.array(data['sl']))
print(np.sort(stats.zscore(np.array(data['sl']))))

# Filter out rows where the absolute value of Z-scores is greater than or equal to 15
# We'll only keep the original rows where Z-scores satisfy this condition
data = data[np.abs(sl_z) < 10]

sw_z = stats.zscore(np.array(data['sw']))
data = data[np.abs(sw_z) < 10]

print('===========')


print(f"sl mean: {data['sl'].mean()}")
print(f"sl std: {np.std(data['sl'])}")

# 4.1
#minmax_scaled = MinMaxScaler().fit_transform(data)

# 4.2
#sd_scaled = StandardScaler().fit_transform(data)

# 4.3
#pca = PCA()
#principal_components = pca.fit_transform(data)

# 4.4


# Plot the pairplot with the filtered data and "species" as hue
# sns.pairplot(data, hue="species")
# plt.show()