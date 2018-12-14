#!/usr/bin/env python
# coding: utf-8

# # Coursework 2

# ## Import mat

# ## While evaluating (testing) your algorithms, you should not consider images of your current query identity taken from the same camera. For example, when you create ranklist for the first query image (index 22, label 3, camera 1, name "1_003_1_02.png"), you should not include images with indexes 21, 23, 24 in this ranking list, as these are images of the same person (label 3) captured by the camera with index 1.

# In[1]:


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

import sys
print(sys.version)


# In[2]:


print(train_idxs.shape)


# # Import JSON file (14096 x 2048)

# In[3]:


import json
import numpy as np
with open('feature_data.json', 'r') as f:
    features = json.load(f)
    
feature = np.asarray(features)


# # Training & Testing & Validating set

# # Training set

# In[4]:


fea_train = []
lbl_train = []

for i in train_idxs:
    fea_train.append(feature[i-1])
    lbl_train.append(labels[i-1])

fea_train = np.asarray(fea_train)
lbl_train = np.asarray(lbl_train)


# # Query

# In[5]:


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


# # Gallery

# In[6]:


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


# # Training & Validation

# In[8]:


from sklearn.utils import shuffle
from tqdm import tqdm
train_unique_labels = np.unique(lbl_train)

# pick 100 of people and move all pictures of those people to your validation set
shuffle_lbl = shuffle(train_unique_labels)[:100] 

val = np.zeros((1,2049))
train = np.vstack((fea_train.T, lbl_train)).T

for identity in tqdm( shuffle_lbl ):
    #Select validation set from training data
    validation = train[ np.where(train[:,-1]==identity)]
    
    val = np.vstack( ( val, validation ) )
    
    train = train[ np.where( train[ :, -1 ] != identity )]
val = np.delete(val, (0), axis=0)

# # Large Margin Nearest Neighbor (LMNN)

# In[11]:


from metric_learn import LMNN
import time
ts = time.time()
X = train[:,:2048]
y = train[:,-1]

lmnn = LMNN(k=5, learn_rate=1e-6)
lmnn.fit(X, y)
te = time.time()
print('Time: %d s'%te-ts)


# In[12]:


print('done')


# In[17]:


q_transform = lmnn.transform(query_feature)
g_transform = lmnn.transform(gallery_feature)


# In[19]:


print(query_feature.shape)
print(q_transform.shape)


# # Combine lbl and feature

# In[20]:


q = q_transform.T
g = g_transform.T
#feature, camid, lbl
#0:2047 2048     2049
q_combine = (np.vstack((q,query_camid,query_lbl))).T

#feature, camid, lbl

g_combine = (np.vstack( ( g, gallery_camid, gallery_lbl ))).T


# In[21]:


print(q.shape)
print(query_camid.shape)
print(query_lbl.shape)
print(q_combine.T.shape)

print(g.shape)
print(gallery_camid.shape)
print(gallery_lbl.shape)
print(g_combine.shape)


# In[22]:


print( 'Query Augmented: {}'.format( q_combine.shape ) )
print( 'Gallery Augmented: {}'.format( g_combine.shape ) )
print(q_combine[2])
print(g_combine[2])


# In[23]:


print(lbl_train.shape)
print(train.shape)

print(np.where(train == 1467))


# # NN

# In[25]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


X_train = g_combine[:,:-2]
y_train = g_combine[:,-1]
classifier = NearestNeighbors(n_neighbors=20, metric = 'euclidean')
classifier.fit(X_train, y_train)

query_rank_list_20 = []

acc = []

for i in tqdm(range(q_combine.shape[0])):
    
    query_lbl = q_combine[i,-1].astype(int)
    
    
    X_test = q_combine[i,:-2].reshape(1,-1)
    dist, index = classifier.kneighbors(X_test)
    index = index.flatten()
    ii = 0
    rank_list = []
    for j in index:
        if len(rank_list) < 15:
            if g_combine[j,-1] !=  q_combine[i,-1] or g_combine[j,-2] !=  q_combine[i,-2]:
                rank_list.append(g_combine[j,-1].astype(int) == query_lbl)
    query_rank_list_20.append(rank_list)

query_rank_list_20 = np.asarray(query_rank_list_20)
np.savetxt( 'lmnn_ranklist.csv', query_rank_list_20, delimiter= ',' )


# In[26]:


def mAP():
    average_precision=[]
    ranklist = np.loadtxt(open("baseline_ranklist.csv", "rb"), delimiter=",")
    for i in range(len(ranklist)):
        precision=[]
        recall=[]
        precisions=[]
        s=sum(ranklist[i,:])
        for j in range(1,ranklist.shape[1]):
            precision.append(sum(ranklist[i,:j]/j))
            recall.append(sum(ranklist[i,:j]/s))
            if recall[j-1] == 1:
                  break

        u=[]
        indices=[]
        recall=np.array(recall)
        precision=np.array(precision)
        u, indices=np.unique(recall,return_index=True)

        precisions=precision[indices]
        precisions=precisions[precisions!=0]
        average_precision.append(np.mean(precisions))
    average_precision = np.nan_to_num(average_precision)
    print('mAP :{}'.format(np.mean(average_precision[1:])))


# In[27]:


rank1 = query_rank_list_20.T[0].T
rank5  = query_rank_list_20.T[:5].T
rank10 = query_rank_list_20.T[:10].T

cmc1  = rank1
cmc5  = np.sum(rank5, axis = 1) > 0
cmc10 = np.sum(rank10, axis = 1) > 0

print( 'rank 1: {}%'.format(np.sum(cmc1)/cmc1.shape[0]* 100))
print( 'rank 5: {}%'.format(np.sum(cmc5)/cmc5.shape[0]* 100 ))
print( 'rank 10: {}%'.format(np.sum(cmc10)/cmc10.shape[0]*100))

mAP()


# In[ ]:




