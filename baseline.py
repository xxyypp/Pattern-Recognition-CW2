'''Import'''
from scipy.io import loadmat
#contains indexes of images that can be used for training and validation (Training Validating)   (7368,) 7368 images  (7-10) per person
train_idxs = loadmat('./PR_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()

#specifies whether image was taken from camera 1 or camera 2.    
camId = loadmat('./PR_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()

# specifies correspondences between names of files in images_cuhk03 and their indexes
filelist = loadmat('./PR_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()

#specifies indexes of the part of the dataset from which you compose your ranklists during testing phase (Testing)
gallery_idx = loadmat('./PR_data/cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()

#contains ground truths for each image                                                           (14096,) 对应着train_idxs的image label
labels = loadmat('./PR_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()

#contains indexes of query images (Testing)
query_idx = loadmat('./PR_data/cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()

import json
import numpy as np
with open('feature_data.json', 'r') as f:
    features = json.load(f)
    
feature = np.asarray(features)

# Training set
fea_train = []
lbl_train = []

for i in train_idxs:
    fea_train.append(feature[i-1])
    lbl_train.append(labels[i-1])

fea_train = np.asarray(fea_train)
lbl_train = np.asarray(lbl_train)

# Query set
query_feature = []
query_lbl = []
query_camid = []

for i in query_idx:
    query_feature.append(feature[i-1])
    query_lbl.append(labels[i-1])
    query_camid.append(camId[i-1])

query_feature = np.asarray(query_feature)
query_lbl = np.asarray(query_lbl)
query_camid = np.asarray(query_camid)

#Gallery
gallery_feature = []
gallery_lbl = []
gallery_camid = []

for i in gallery_idx:
    gallery_feature.append(feature[i-1])
    gallery_lbl.append(labels[i-1])
    gallery_camid.append(camId[i-1])

gallery_feature = np.asarray(gallery_feature)
gallery_lbl = np.asarray(gallery_lbl)
gallery_camid = np.asarray(gallery_camid)

# Combine lbl and feature
q = query_feature.T

#feature, camid, lbl
#0:2047 2048     2049
q_combine = (np.vstack((q,query_camid,query_lbl))).T

#feature, camid, lbl
g = gallery_feature.T
g_combine = (np.vstack( ( g, gallery_camid, gallery_lbl ))).T

#NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm_notebook
knn_n_neighbour = 20
knn_metric = 'euclidean'

query_rank_list_20 = []
acc = []

for i in range(q_combine.shape[0]):
    query_lbl = q_combine[i,-1].astype(int)
    
    #Delete same camid and label
    gallery_del = g_combine[~np.logical_and((g_combine[:,-1] == q_combine[i,-1]),
                                            (g_combine[:,-2] == q_combine[i,-2]))]
    
    
    y_train = gallery_del[:,-1]
    X_train = gallery_del[:,:-2]
    classifier = NearestNeighbors(n_neighbors=knn_n_neighbour, metric = knn_metric)
    classifier.fit(X_train, y_train)
    
    #classifier_ = KNeighborsClassifier(n_neighbors=knn_n_neighbour, metric = knn_metric)
    #classifier_.fit(X_train, y_train)
    #acc_ = classifier_.score(query_feature,query_lbl)
    #acc.append([acc])
    
    X_test = q_combine[i,:-2].reshape(1,-1)
    dist, index = classifier.kneighbors(X_test)
    rank_list = [gallery_del[i,-1].astype(int) == query_lbl for i in index.flatten()]
    query_rank_list_20.append(rank_list)

query_rank_list_20 = np.asarray(query_rank_list_20)
