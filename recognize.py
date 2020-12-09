#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout, Input
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
import cv2
import os
from keras.models import load_model


# In[2]:


directory = './Dataset'

possible_labels = os.listdir(directory)

possible_dict = {}
possible_dict_reverse = {}

for i in range(len(possible_labels)):
    possible_dict[possible_labels[i]] = i
    possible_dict_reverse[i] = possible_labels[i]


# In[3]:


model = load_model("best_model")


# In[ ]:

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1200, 900))
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, (700, 300), (924, 524), (0, 255, 0))
    hand = img[300: 524, 700:924, :]
    cv2.imshow("hand", hand)

    hand = hand.reshape((1, 224, 224, 3))
    pred = model.predict(hand)
    label = np.argmax(pred)
    cv2.putText(img, "Prediction is: " +
                str(possible_dict_reverse[label]), (700, 300-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("live", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:

# In[ ]:
# In[ ]:
