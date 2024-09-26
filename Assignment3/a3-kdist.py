# Plot k-distances
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors

def k_distances2(X,n):
    neighbors = NearestNeighbors(n_neighbors=n)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    return distances, indices

# TODO: add your parameters here.
# data -- your normalized dataset
# k    -- k-th neighbour. By default, k=count(features)+1
distances, indices = k_distances2(data,5)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()