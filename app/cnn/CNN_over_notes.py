#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import time

import matplotlib
import keras
import pandas as pd
import swifter
import numpy as np
import json
import pickle

from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
import tensorflow
from IPython.display import display

from app.features.MidiData import MidiData

DTYPE = {
    'canonical_composer': 'category',
    'canonical_title': 'object',
    'split': 'category',
    'year': 'int64',
    'midi_filename': 'object',
    'audio_filename': 'object',
    'duration': 'float64'
}

FEATURES = 1


def number_of_classes(classes):
    return len(classes.canonical_composer.cat.categories)


def get_midi_data(filename):
    return MidiData(os.path.join("../../data/maestro-v2.0.0/", filename))


# In[7]:


data = pd.read_json('../../data/maestro-v2.0.0/maestro-v2.0.0.json')
dataset = data.astype(DTYPE)
start_time = time.time()


# In[18]:


dataset['midi_data'] = dataset['midi_filename'].swifter.progress_bar().allow_dask_on_strings().apply(get_midi_data)


# In[20]:


print(dataset)


# In[19]:


pickle.dump(dataset, open('../../dataset.pickle', 'wb'))


# In[10]:


with open('../../dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)


# In[22]:


current_time = time.time()
print(f'Elapsed time: {(current_time - start_time) / 60} minute(s)')


# In[34]:


def get_notes(midi_data):
    return np.asarray([n[0] for n in midi_data.notes.data])


dataset['raw_notes'] = dataset.midi_data.swifter.apply(get_notes)
dataset['raw_notes_count'] = dataset.raw_notes.swifter.apply(lambda a: len(a))


# In[118]:


def code_to_label(code):
    return dataset.canonical_composer.cat.categories[code]


# In[36]:


notes_count_mean = dataset.raw_notes_count.mean()
inf_bound = notes_count_mean - 1 * dataset.raw_notes_count.std()
sup_bound = notes_count_mean + 1 * dataset.raw_notes_count.std()

reduced_dataset = dataset[
    (dataset.raw_notes_count >= inf_bound) & (dataset.raw_notes_count <= sup_bound)
].copy()
display(reduced_dataset[["raw_notes_count"]].describe())

stats = pd.DataFrame(
    {"raw_notes_count": dataset.raw_notes_count.sort_values(ignore_index=True),}
)
plot = stats.plot(style={"inf_bound": "--", "sup_bound": "--"})
plot.set(xlabel='Sample index', ylabel='Notes count')

mean_index = stats.index[stats.raw_notes_count >= notes_count_mean][0]
inf_bound_index = stats.index[stats.raw_notes_count >= inf_bound][0]
sup_bound_index = stats.index[stats.raw_notes_count >= sup_bound][0]
mean_line = plot.axvline(x=mean_index, c="orange", label="mean")

# Draw separator lines for selected subset
inf_bound_line = plot.axvline(x=inf_bound_index, c="green", ls="--", label="inf_bound")
sup_bound_line = plot.axvline(x=sup_bound_index, c="red", ls="--", label="sup_bound")
plot.legend(handles=[mean_line, inf_bound_line, sup_bound_line])

# Extract stats for the notes counts
display(dataset[["raw_notes_count"]].describe())
display(stats)


# In[64]:


max_notes_count = reduced_dataset.raw_notes_count.max()
min_notes_count = reduced_dataset.raw_notes_count.min()

def pad_array(a):
    return a[0:min_notes_count]

# Pad notes array to max length
reduced_dataset["padded_notes"] = reduced_dataset.raw_notes.swifter.apply(pad_array)


# In[ ]:


# Shuffle dataset
from sklearn.utils import shuffle

reduced_dataset = shuffle(reduced_dataset)


# In[114]:


def format_data(data_to_format):
    train = data_to_format[data_to_format.split == 'train']
    test = data_to_format[data_to_format.split == 'test']
    validation = data_to_format[data_to_format.split == 'validation']
    x_split_train = np.vstack(train['padded_notes']).reshape(len(train.index), -1, FEATURES)
    y_split_train = np.asarray(train['canonical_composer'].cat.codes)
    x_split_test = np.vstack(test['padded_notes']).reshape(len(test.index), -1, FEATURES)
    y_split_test = np.asarray(test['canonical_composer'].cat.codes)
    x_split_validation = np.vstack(validation['padded_notes']).reshape(len(validation.index), -1, FEATURES)
    y_split_validation = np.asarray(validation['canonical_composer'].cat.codes)

    return x_split_train, y_split_train, x_split_test, y_split_test, x_split_validation, y_split_validation


NUM_CLASSES = number_of_classes(dataset)


# In[126]:


x_train, y_train, x_test, y_test, x_validation, y_validation = format_data(reduced_dataset)
print(x_train[0].shape)
print(reduced_dataset['split'])
print(reduced_dataset['padded_notes'][0])
print(np.concatenate(x_train[0], axis=0))
print(reduced_dataset['canonical_composer'][0])
print((reduced_dataset['padded_notes'][0] == np.concatenate(x_train[0], axis=0)).all())
print(code_to_label(y_train[0]))


# In[116]:


def build_model(input_shape, output_shape):
    print(f'Input shape : {input_shape}')
    print(f'Output shape : {output_shape}')

    model_to_build = keras.Sequential(
        [
            keras.layers.Conv1D(filters=512, kernel_size=9, activation='relu', input_shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(output_shape),
        ]
    )
    model_to_build.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_to_build.summary()
    return model_to_build


# In[117]:


model = build_model(x_train[0].shape, NUM_CLASSES)


# In[43]:


keras.utils.plot_model(model, show_shapes=True)


# In[8]:


checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(), "models", "best.hdf5"),
                                             monitor='val_accuracy', verbose=2,
                                             save_best_only=True, mode='max')
tb_callback = tensorflow.keras.callbacks.TensorBoard(log_dir='./logs/',
                                             profile_batch=(0, 32))

history = model.fit(x_train, y_train, batch_size=8, epochs=10,
#                    callbacks=[checkpoint],
                    validation_data=(x_test, y_test), use_multiprocessing=True)
loss, accuracy = model.evaluate(x_test, y_test, steps=10, use_multiprocessing=True)
print(f'Model has achieved {accuracy}% of accuracy with {loss} loss')


# In[127]:


pd.DataFrame(history.history)[["loss", "val_loss"]].plot().set(xlabel="Epoch", ylabel="Loss")
pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot().set(xlabel="Epoch", ylabel="Accuracy")


# In[97]:


idx = np.random.choice(len(x_test))
sample, sample_label = x_test[idx], y_test[idx]

test_model = build_model(x_test[0].shape, NUM_CLASSES)
test_model.set_weights(model.get_weights())
result = tensorflow.argmax(test_model.predict_on_batch(tensorflow.expand_dims(sample, 0)), axis=1)
print(f'Predicted result is: {code_to_label(result)}, target result is: {code_to_label(sample_label)}, idx: {idx}')

