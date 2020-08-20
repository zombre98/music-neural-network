#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas
import swifter

tf.debugging.set_log_device_placement(False)

from app.features.MidiData import MidiData

DATASET_DIR = "../../data/maestro-v2.0.0"


# In[2]:


# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[2]:


dtype = {
    "canonical_composer": "category",
    "canonical_title": "object",
    "split": "category",
    "year": "int64",
    "midi_filename": "object",
    "audio_filename": "object",
    "duration": "float64",
}
dataset = pandas.read_json(os.path.join(DATASET_DIR, "maestro-v2.0.0.json"))
dataset = dataset.astype(dtype)


# In[3]:


def get_midi_data(filename):
    return MidiData(os.path.join(DATASET_DIR, filename))


dataset["midi_data"] = (
    dataset.midi_filename.swifter.progress_bar()
    .allow_dask_on_strings()
    .apply(get_midi_data)
)


# In[3]:


def pad_array(a):
    return np.pad(a, (0, max_notes_count - a.shape[0]), mode="constant")


def code_to_label(code):
    return dataset.canonical_composer.cat.categories[code][0]


def get_notes(midi_data):
    return midi_data.notes.get().note.to_numpy()


# In[4]:


dataset["raw_notes"] = dataset.midi_data.swifter.apply(get_notes)
dataset["raw_notes_count"] = dataset.raw_notes.swifter.apply(lambda a: len(a))
# max_notes_count = dataset.raw_notes_count.max()
# dataset["padded_notes"] = dataset.raw_notes.swifter.apply(pad_array)


# In[4]:


import pickle

with open('./dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)


# In[5]:


# Compute stats for notes counts
notes_count_mean = dataset.raw_notes_count.mean()
inf_bound = notes_count_mean - 1 * dataset.raw_notes_count.std()
sup_bound = notes_count_mean + 1 * dataset.raw_notes_count.std()

# Graph notes counts wrt indexes
stats = pandas.DataFrame(
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


# In[6]:


# Subset dataset based on computed boundaries
reduced_dataset = dataset[
    (dataset.raw_notes_count >= inf_bound) & (dataset.raw_notes_count <= sup_bound)
].copy()
display(reduced_dataset[["raw_notes_count"]].describe())
reduced_dataset


# In[7]:


# Compute max count to pad array to max length
max_notes_count = reduced_dataset.raw_notes_count.max()
min_notes_count = reduced_dataset.raw_notes_count.min()

# Pad notes array to max length
reduced_dataset["padded_notes"] = reduced_dataset.raw_notes.swifter.apply(pad_array)
display(reduced_dataset.padded_notes.iloc[0])
pandas.DataFrame(
    {"min_notes_count": [min_notes_count], "max_notes_count": max_notes_count}
)


# In[8]:


# Shuffle dataset
from sklearn.utils import shuffle

reduced_dataset = shuffle(reduced_dataset)


# In[9]:


timesteps = max_notes_count
features = 1
output_size = len(reduced_dataset.canonical_composer.cat.categories)


# In[10]:


train = reduced_dataset[reduced_dataset.split == "train"]
test = reduced_dataset[reduced_dataset.split == "test"]
validation = reduced_dataset[reduced_dataset.split == "validation"]
x_train = np.vstack(train["padded_notes"]).reshape(len(train.index), -1, features)
y_train = np.asarray(train["canonical_composer"].cat.codes)
x_test = np.vstack(test["padded_notes"]).reshape(len(test.index), -1, features)
y_test = np.asarray(test["canonical_composer"].cat.codes)
x_validation = np.vstack(validation["padded_notes"]).reshape(len(validation.index), -1, features)
y_validation = np.asarray(validation["canonical_composer"].cat.codes)

print("dataset created:")
print(
    f"{x_train.shape[0]*100/(x_train.shape[0]+x_test.shape[0]):.0f}% of the data for training"
)
print(f"train: {x_train.shape}, {y_train.shape}")
print(f"test:  {x_test.shape}, {y_test.shape}")


# In[11]:


def build_model():
    mask_layer = keras.layers.Masking(mask_value=0, input_shape=(timesteps, features))
    deep_lstm_layer = keras.layers.LSTM(32, return_sequences=True)
    lstm_layer = keras.layers.LSTM(32)
    return keras.models.Sequential(
        [
            mask_layer,
            deep_lstm_layer,
            lstm_layer,
            keras.layers.Dense(output_size),
        ]
    )


# In[12]:


model = build_model()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
)

display(model.summary())

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=12)
checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(os.getcwd(), "saved_models", "best.hdf5"),
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

last_history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=16,
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping, checkpoint],
)


# In[13]:


get_ipython().run_line_magic('tensorboard', '--host 192.168.1.23 --logdir ./logs/fit')

# pandas.DataFrame(last_history.history)[["loss", "val_loss"]].plot().set(xlabel="Epoch", ylabel="Loss")
# pandas.DataFrame(last_history.history)[["accuracy", "val_accuracy"]].plot().set(xlabel="Epoch", ylabel="Accuracy")


# In[17]:


idx = np.random.choice(len(x_validation))
sample, sample_label = x_validation[idx], y_validation[idx]

test_model = build_model()
test_model.load_weights(os.path.join(os.getcwd(), "saved_models", "best.hdf5"))
result = tf.argmax(test_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
print(
    f"Predicted result is: {code_to_label(result.numpy())}, target result is: {code_to_label([sample_label])}"
)

