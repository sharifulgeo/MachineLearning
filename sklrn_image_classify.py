# simple k means clustering
from sklearn import cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os

first = plt.imread(r"C:\Users\Winrock\Desktop\luca\prcA.jpg")
dims = np.shape(first)
pixel_matrix = np.reshape(first, (dims[0] * dims[1], dims[2]))
kmeans = cluster.KMeans(2)
#sklearn.
clustered = kmeans.fit_predict(pixel_matrix)

dims = np.shape(first)
clustered_img = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img)