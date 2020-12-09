#!/usr/bin/env python
# coding: utf-8

# In[17]:


from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout, Input, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.applications import ResNet50

img_size = 224

# In[20]:

X = np.load("data.npy").reshape(-1, img_size, img_size, 3)
Y = np.load("label.npy")

# In[ ]:


print(X.shape, Y.shape)


# In[22]:


def drawImg(img):
    plt.style.use("seaborn")
    plt.imshow(img, cmap='gray')
    plt.grid(False)
    plt.show()


# In[23]:
# drawImg(X[1])

# In[25]:


model = Sequential()
model.add(Input(shape=(224, 224, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(26, activation='softmax'))


# In[26]:


model.summary()


# In[27]:


model.compile(loss="categorical_crossentropy",
              optimizer='SGD', metrics='accuracy')


# In[28]:


checkpoint = ModelCheckpoint(
    'best_model1', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')


# In[30]:

history1 = model.fit(X, Y, epochs=10, batch_size=50,
                     shuffle=True, validation_split=0.1, callbacks=[checkpoint])


plt.style.use("seaborn")
plt.plot(history1.history['loss'], 'r', label='training loss')
plt.plot(history1.history['val_loss'], label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('LOSS')
plt.legend()
plt.show()

plt.plot(history1.history['accuracy'], 'r', label='training accuracy')
plt.plot(history1.history['val_accuracy'], label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
# In[ ]:

# test_data = np.load("data_test.npy")
# test_label = np.load("label_test.npy")

# pred = model.predict(test_data.reshape(-1,200,200,3))

# final_pred = np.zeros((pred.shape[0], 26))
# for i in range(pred.shape[0]):
#     x = np.argmax(pred[i])
#     final_pred[i][x] = 1

# count = 0
# for i in range(final_pred.shape[0]):
#     if sum(final_pred[i] == test_label[i]) == 26:
#         count+=1

# print(count/test_data.shape[0])


# In[ ]:
model.save("latest_model")
