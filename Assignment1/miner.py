import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



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

print('===========')

# 4.1
# Apply MinMaxScaler to each column and store the result in a new column
data[['pl','pw','sl','sw']] = MinMaxScaler((0, 1)).fit_transform(data[['pl','pw','sl','sw']])

print(f"minmax_scaled: sl mean: {data['sl'].mean()}")
print(f"minmax_scaled:sl std: {np.std(data['sl'])}")

print('===========')


# 4.2
data[['pl','pw','sl','sw']] = StandardScaler().fit_transform(data[['pl','pw','sl','sw']])
print(f"sd_scaled: sl mean: {data['sl'].mean()}")
print(f"sd_scaled: sl std: {np.std(data['sl'])}")

print('===========')


# 4.3
# Apply PCA with N principal components
pca = PCA()
principal_components = pca.fit_transform(data[['pl', 'pw', 'sl', 'sw']])

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
n=-1
for i in range(0,5):
    if pca.explained_variance_ratio_.cumsum()[i] >= 0.95:
        n = i+1
        break
print(f"95% requires {n} components")


pca = PCA(n_components=2)
principal_components = pca.fit_transform(data[['pl', 'pw', 'sl', 'sw']])

# Check how many components were selected
#print(f"Number of components selected: {pca.n_components}")

print('===========')

# 4.4
#Create a DataFrame of the PCA components

r = pd.DataFrame(pca.components_, columns = ["Sepal L", "Sepal W",
"Petal L" , "Petal W"], index =['PC 1', 'PC 2']).abs().mean(axis = 0)
print(pca.components_)
print(pca.explained_variance_ratio_)

print('===========')

# Check the absolute contribution of each attribute to the principal components
#mean_contribution = pca_df.abs().mean(axis=0)
#print(mean_contribution)

# 4.5
data[['pl']] = MinMaxScaler((0, 100)).fit_transform(data[['pl']])
pca_mm = PCA()
principal_components = pca.fit_transform(data[['pl', 'pw', 'sl', 'sw']])
print(pca.components_)
print(pca.explained_variance_ratio_)


print('===========')

#4.6
data[['pl']] = MinMaxScaler((0, 1)).fit_transform(data[['pl']])
data['pl'][0] = 5000
data[['pl']] = MinMaxScaler((0, 1)).fit_transform(data[['pl']])
pca_mm = PCA()
principal_components = pca.fit_transform(data[['pl', 'pw', 'sl', 'sw']])
print(pca.components_)
print(pca.explained_variance_ratio_)


print('===========')



def print_repeated_identifiers(m, data):
    print(f"{m}: Repeats: {data['id'].duplicated().any()}")

# Sampling

# 5.1
sample = data.sample(n = 150)
print_repeated_identifiers("Sampling", sample)
print(sample.value_counts("species"))

print('===========')

# 5.2
sample = data.sample(n = 150, replace=True)
print_repeated_identifiers("Bootsraping", sample)
print(sample.value_counts("species"))

print('===========')

# 5.3
sample = data.groupby('species' , group_keys = False ).apply(lambda x : x.sample(frac =0.5))
print_repeated_identifiers("Stratified", sample)
print(sample.value_counts('species'))

print('===========')

# 5.4
sample = data.groupby('species' , group_keys = False).apply(lambda x : x.sample(50))
print_repeated_identifiers("Stratified ver. 2", sample)
#print(sample.value_counts('species'))
# Plot the pairplot with the filtered data and "species" as hue
#sns.pairplot(data, hue="species")
#plt.show()