#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils


# In[2]:


directory = '.\Dataset'

possible_labels = os.listdir(directory)

possible_dict = {}
possible_dict_reverse = {}

for i in range(len(possible_labels)):
    possible_dict[possible_labels[i]] = i
    possible_dict_reverse[i] = possible_labels[i]


# In[3]:


# labels = []
# data = []
# for i in possible_labels:
#     label = i[0].lower()
#     print(os.path.join(directory, i))
#     image = cv2.imread(os.path.join(directory, i))
#     image = cv2.resize(image, (200,200))
#     data.append(image)
#     labels.append(ord(label)-ord('a'))


# In[4]:


labels = []
data = []
for i in possible_labels:
    print("working on :", i)
    images_name = os.listdir(os.path.join(directory, i))
    for j in images_name:
        image = cv2.imread(os.path.join(directory, i, j))
        image = cv2.resize(image, (224,224))
        data.append(image)
        labels.append(possible_dict[i])


# In[5]:


from sklearn.decomposition import PCA


# In[6]:


data = np.array(data)
labels = np_utils.to_categorical(np.array(labels), dtype = int)


# In[9]:


temp = np.hstack((data, labels))


# In[10]:


np.random.shuffle(temp)


# In[11]:


print(temp.shape)


# In[12]:


# data1 = temp[:,:-26]
# label1 = temp[:,-26:]


# In[13]:


np.save("data", temp[:,:-26])


# In[14]:


np.save("label", temp[:,-26:])


# In[15]:


print(labels)


# In[ ]:




