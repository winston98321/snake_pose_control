"""
Created on Tue Jun 14 21:15:03 2022

@author: PO KAI
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import pandas as pd
import pickle # Object serialization.
from sklearn.model_selection import train_test_split

 

def load_dataset(csv_data):
    df = pd.read_csv(csv_data)
    features = df.drop('class', axis=1) # Features, drop the colum 1 of 'class'.
    target_value = df['class']          # target value.

    x_train, x_test, y_train, y_test = train_test_split(features, target_value, test_size=0.3, random_state=1234)

    return x_train, x_test, y_train, y_test
 


if __name__ == '__main__':
    
    dataset_csv_file = './dataset/pushup_num.csv'
    model_weights = './model_weights/pushup.pkl'
   
    x_train = load_dataset(csv_data=dataset_csv_file)[0]
    y_train = load_dataset(csv_data=dataset_csv_file)[2]
    x_test = load_dataset(csv_data=dataset_csv_file)[1]
    y_test = load_dataset(csv_data=dataset_csv_file)[3]
   
train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))

ds_train = train_ds.shuffle(buffer_size=100)

ds_train = ds_train.batch(batch_size=2)
ds_train = ds_train.prefetch(buffer_size=1000)
    
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='sigmoid', 
                          name='fc1',input_shape=(132,)),
    tf.keras.layers.Dense(128,name='l2', activation='relu'),
    tf.keras.layers.Dense(64,name='l3', activation='relu'),
    tf.keras.layers.Dense(32,name='l4', activation='relu'),
    tf.keras.layers.Dense(2, name='fc2', activation='softmax')])

model.summary()





model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])




num_epochs = 10
training_size = 20
batch_size = 4
steps_per_epoch = np.ceil(training_size / batch_size)




history = model.fit(ds_train,
                    batch_size = 4,epochs=num_epochs,
                         verbose=2)




hist = history.history

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(hist['loss'], lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
#plt.savefig('ch13-cls-learning-curve.pdf')

plt.show()



results = model.evaluate(ds_test.batch(50), verbose=0)
print('Test loss: {:.4f}   Test Acc.: {:.4f}'.format(*results))


# ### Saving and reloading the trained model



model.save('pushup-classifier.h5', 
                overwrite=True,
                include_optimizer=True,
                save_format='h5')




model_new = tf.keras.models.load_model('pushup-classifier.h5')

model_new.summary()




results = model_new.evaluate(ds_test.batch(50), verbose=0)
print('Test loss: {:.4f}   Test Acc.: {:.4f}'.format(*results))




labels_train = []
for i,item in enumerate(train_ds):
    labels_train.append(item[1].numpy())
    
labels_test = []
for i,item in enumerate(ds_test):
    labels_test.append(item[1].numpy())
print('Training Set: ',len(labels_train), 'Test Set: ', len(labels_test))



model_new.to_json()
