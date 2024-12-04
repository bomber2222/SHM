#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:23:05 2024

@author: gabriel


I have understood (2.1 UL) that random ICs 
and windowing dont work. In this code i have 
implemented no windowing and norandom ICs: the 
clustering is done on the input response of the 
system on T_end long response
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, homogeneity_completeness_v_measure

freq = 50 ; T_end = 15

def h_to_label(h):
    
    global N_classes
    
    ## Here is possible to choose in how many classes 
    ## of health the system is classified
    
    # if h>=0.8:
    #     label=4
    # if h>=0.6 and h<0.8:
    #     label=3
    # if h>=0.4 and h<0.6:
    #     label=2
    # if h>=0.2 and h<0.4:
    #     label=1
    # if h<0.2:
    #     label=0
    # N_classes =5
     
    if h>=0.7:
        label=2
        
    if h>=0.3 and h<0.7:
        label=1
    
    if h<0.3:
        label=0
    
    N_classes=3
        
    return label

def motion(h, training):
    k=100*h; m=5  
    k+=1
    
    time = np.linspace(0,T_end,T_end*freq+1); Dt = time[1]
    u = np.zeros(len(time))
    u[:int(0.1*freq)]=10
    h_damp = 0.1
    c = h_damp*2*np.sqrt(k*m)
    
    A = np.zeros((2,2)); B = np.zeros((2,1))
    
    A[0,1]=Dt; A[1,1]=-c/m*Dt; A[1,0] = -k*Dt/m
    B[1,0]=Dt/m
    
    # x = np.random.uniform(-10,10,(2,1))
    x = np.array([[0],[0]],dtype=np.float32)
    y = []
    k=0
    
    for j in time[:-1]:
        y.append(x[0,0])
        x += A@x + B*u[k]
        k+=1
    
    if training:
        return np.array(y).reshape(1,len(y)), np.array([[h_to_label(h)]])
    
    else:
        return y, time[:len(y)]
    
def data(N, training):
    
    h_list = np.random.uniform(0,1,(N,1))
    first_time = True
    
    for h in h_list:
        matrix, label = motion(h, training)
        matrix1 = np.concatenate((matrix, label),axis=1)
        
        if first_time:
            X = matrix1.copy()
            first_time = False
        else:
            X = np.concatenate((X,matrix1),axis = 0)
    
    np.random.shuffle(X)
    
    return X

# Step 1: Generate data. Exact labels are known 
# (since the data are simulated) to compare the 
# results of unsupervised learning. 
data_aux = data(1000,True)
X = data_aux[:,:freq]
True_labels = data_aux[:,-1:]
# StandardScaler normalizes the features (mean=0, variance=1), which is often necessary for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Reduce dimension
# Since the data is 50-dimensional (1 second time history is registered),
# we need to reduce the dimension with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=N_classes, n_init='auto')  # Change `n_clusters` to the number of desired clusters
kmeans.fit(X_pca)
# Get the cluster labels
labels_pred = kmeans.labels_.reshape(len(X[:,0]),1)  # Labels for each sample (which cluster it belongs to)

# Step 4: Cleaning data
# The simulations with labels at the corssover 
# between one class and other could compromise the 
# results so it is improtant to filter out the border points
# Get the cluster centroids
centroids = kmeans.cluster_centers_
# Calculate the distance of each point from its assigned centroid
distances = np.linalg.norm(X_pca - centroids[kmeans.labels_], axis=1)
# Define a threshold for "outlier" points (e.g., points far from centroids)
threshold = np.percentile(distances, 80)  # Use the 90th percentile as a threshold (or pick another value)
# Identify points that are farther than the threshold
outliers = distances > threshold
# Remove outlier points
cleaned_data = X_pca[~outliers]
true_labels_2 = True_labels[~outliers]
labels_pred_2 = labels_pred[~outliers]

# Step 5.1: Evaluate the clustering performance
ari_score = adjusted_rand_score(True_labels[:,0], labels_pred[:,0])
print(f"Adjusted Rand Index (ARI): {ari_score:.2f}")
labels = np.concatenate((True_labels,labels_pred), axis=1)

# Step 6.1: Confusion Matrix (Contingency Table)
conf_matrix = confusion_matrix(True_labels[:,0], labels_pred)
print("Confusion Matrix (Contingency Table):")
print(conf_matrix)

print('After filtering out')
# Step 5.2: Evaluate the clustering performance after the filtering
ari_score = adjusted_rand_score(true_labels_2[:,0], labels_pred_2[:,0])
print(f"Adjusted Rand Index (ARI): {ari_score:.2f}")
labels = np.concatenate((True_labels,labels_pred), axis=1)

# Step 6.2: Confusion Matrix (Contingency Table) after the filtering
conf_matrix = confusion_matrix(true_labels_2[:,0], labels_pred_2)
print("Confusion Matrix (Contingency Table):")
print(conf_matrix)

# Step 7: Visualize the result
# Create a figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],c=kmeans.labels_, cmap='viridis', marker='o', label='Data points')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=200, c='red', marker='x', label='Centroids')
plt.scatter(cleaned_data[:, 0], cleaned_data[:, 1], cleaned_data[:, 2], c='black', marker='x', label='Cleaned data')
plt.legend()
plt.title("K-Means Clustering with Outliers Removed")
plt.show()





