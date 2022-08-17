import pandas as pd
import pycmap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture


# Load in the data using pycmap with individual token
api = pycmap.API(token='5b820290-1db0-11ed-be7c-2b540ba2c4c4')       # call api using your unique API key
# Convert data read in to a pandas data frame
data = api.get_dataset('tblHOT_Bottle')
# Look at the keys
data.keys()

cols =['NO2_NO3_bottle_hot', 'PO4_bottle_hot', 'SiO4_bottle_hot', 'potential_temperature_ctd_bottle_hot', 'salinity_bottle_hot', 'oxygen_ctd_bottle_hot', 'chl_bottle_hot']
# Remove data with NaN in columns of interest
for j in np.arange(len(cols)):
    data = data.dropna(subset=[cols[j]])

# Filter for depth less than 300 m
data = data[data.depth<300]
# If we want to grid to every 20 m:
data['binned_depth'] = data['depth']//20*20
data

# Make a profile plot for each nutrient with all the data overlaid
fig, axs=plt.subplots(1,len(cols),figsize=(12, 4))
for j in np.arange(len(cols)):
    axs[j].scatter(data[cols[j]],data['pressure_ctd_bottle_hot'])
    axs[j].set_title(cols[j])
    axs[j].invert_yaxis()
    if j>0: 
        axs[j].set_yticklabels([])
axs[0].set_ylabel('Pressure [dbar]')
plt.show()

# How many individual measurements do we have
len(data)

cols = ['NO2_NO3_bottle_hot', 'PO4_bottle_hot', 'SiO4_bottle_hot', 'potential_temperature_ctd_bottle_hot', 'salinity_bottle_hot', 'oxygen_ctd_bottle_hot', 'binned_depth']
features = np.array(data[cols])
# normalize features
features=features/np.std(features,axis=0)
features2 = np.transpose(features)


K=4 # number of clusters picked from below
gmm = GaussianMixture(n_components=K)
labels_gmm=gmm.fit_predict(features)
x_mean,y_mean=gmm.means_[:,0],gmm.means_[:,1]

fig = plt.figure(figsize=(10,10))
plt.title('Gaussian Mixture Model clustering')
for k in range(K):
    plt.plot(x_mean[k],y_mean[k],'rs',markersize=6)
    plt.annotate(str(k), (x_mean[k],y_mean[k]), fontsize=20)    
plt.show()

BIC=np.array([]);
for K in range(1,50):
    gmm = GaussianMixture(n_components=K)
    gmm.fit_predict(features)
    bic=gmm.bic(features)
    BIC=np.append(BIC,bic)
fig = plt.figure(figsize=(10,10))
plt.title('BIC')
plt.plot(BIC); 
plt.xlabel('Number of classes'); 
plt.ylabel('Bayesian information criterion (BIC)')
plt.show()

