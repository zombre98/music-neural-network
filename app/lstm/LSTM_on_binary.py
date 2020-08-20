#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from app.features.MidiData import MidiData


# In[22]:


import pandas

dtype = {
    'canonical_composer': 'category',
    'canonical_title': 'object',
    'split': 'category',
    'year': 'int64',
    'midi_filename': 'object',
    'audio_filename': 'object',
    'duration': 'float64'
}
dataset = pandas.read_json('../../data/maestro-v2.0.0/maestro-v2.0.0.json')
dataset = dataset.astype(dtype)


# In[23]:


get_ipython().run_cell_magic('time', '', "def get_midi_data(filename):\n    return MidiData(os.path.join(DATASET_DIR, filename))\n\ndef filesize(filename):\n    return os.path.getsize('../../data/maestro-v2.0.0/' + filename)\n\ndef read_file(filename):\n    with open('../../data/maestro-v2.0.0/' + filename, 'rb') as f:\n        return f.read()\n    \ndef bytes_as_array(limit):\n    def with_limit(file):\n        return np.frombuffer(file[0:limit], dtype='int8') / 255.0\n    return with_limit\n\ndef code_to_label(code):\n    return dataset.canonical_composer.cat.categories[code][0]\n\ntesting = pandas.DataFrame(dataset).sample(frac=1).reset_index(drop=True)\ntesting['midi_filesize'] = testing['midi_filename'].apply(filesize)\ntesting['midi_file'] = testing['midi_filename'].apply(read_file)\ntesting['midi_data'] = testing['midi_filename'].swifter.progress_bar().allow_dask_on_strings().apply(get_midi_data)")


# In[20]:


input_dim = 1
# input_dim = 28

units = 128
output_size = len(testing.canonical_composer.cat.categories)
print(f'output_size: {output_size}')
# output_size = 10

# Build the RNN model
def build_model():
    lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    return keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )


# In[54]:


train = testing[testing.split == 'train']
test = testing[testing.split == 'test']
x_train = np.vstack(train['bytes_features']).reshape(len(train.index), -1, 1)
y_train = np.asarray(train['canonical_composer'].cat.codes)
x_test = np.vstack(test['bytes_features']).reshape(len(test.index), -1, 1)
y_test = np.asarray(test['canonical_composer'].cat.codes)

print('dataset created:')
print(f'train: {x_train.shape}, {y_train.shape}')
print(f'test:  {x_test.shape}, {y_test.shape}')

model = build_model()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()


checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(os.getcwd(), 'saved_models', 'best.hdf5'),
    monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='max'
)


model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=20, callbacks=[checkpoint])


# In[71]:


idx = np.random.choice(len(x_train))
sample, sample_label = x_train[idx], y_train[idx]

test_model = build_model()
test_model.load_weights(os.path.join(os.getcwd(), 'saved_models', 'best.hdf5'))
result = tf.argmax(test_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
print(f'Predicted result is: {code_to_label(result.numpy())}, target result is: {code_to_label([sample_label])}')


# In[41]:


from sklearn.model_selection import KFold

save_dir = './k_fold_models/'

def get_model_name(k):
    return 'model_' + str(k) + '.h5'

x_data = np.vstack(testing['bytes_features']).reshape(len(testing.index), -1, 1)
y_data = np.asarray(testing['canonical_composer'].cat.codes)

kf = KFold(5)
fold_no = 1

k_fold_model = build_model()
k_fold_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"]
)
    
for train_index, test_index in kf.split(x_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    
    checkpoint = keras.callbacks.ModelCheckpoint(save_dir + get_model_name(fold_no),
        monitor='val_accuracy', verbose=1,
        save_best_only=True, mode='max')

    k_fold_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=4, epochs=10, callbacks=[checkpoint])
    
    fold_no += 1


# In[53]:


idx = np.random.choice(len(x_train))
sample, sample_label = x_train[idx], y_train[idx]

test_model = build_model()
test_model.set_weights(k_fold_model.get_weights())
result = tf.argmax(test_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
print(f'Predicted result is: {code_to_label(result.numpy())}, target result is: {code_to_label([sample_label])}')

for model_file in os.listdir(os.path.join(os.getcwd(), 'k_fold_models')):
    test_model.load_weights(os.path.join(os.getcwd(), 'k_fold_models', model_file))
    result = tf.argmax(test_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
    print(f'{model_file} predicted result is: {code_to_label(result.numpy())}')

