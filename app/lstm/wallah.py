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

def pad_array(a):
    return np.pad(a, (0, max_notes_count - a.shape[0]), mode="constant")


def code_to_label(code):
    return dataset.canonical_composer.cat.categories[code][0]


def get_notes(midi_data):
    return midi_data.notes.get().note.to_numpy()

import pickle

try:
    with open('./dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)
except OSError as err:
    with open('./dataset.pickle', 'wb') as f:
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
        def get_midi_data(filename):
            return MidiData(os.path.join(DATASET_DIR, filename))

        dataset["midi_data"] = (
            dataset.midi_filename.swifter.progress_bar()
            .allow_dask_on_strings()
            .apply(get_midi_data)
        )

        dataset["raw_notes"] = dataset.midi_data.swifter.apply(get_notes)
        dataset["raw_notes_count"] = dataset.raw_notes.swifter.apply(lambda a: len(a))

        pickle.dump(dataset, f)

# Compute stats for notes counts
notes_count_mean = dataset.raw_notes_count.mean()
inf_bound = notes_count_mean - 1 * dataset.raw_notes_count.std()
sup_bound = notes_count_mean + 1 * dataset.raw_notes_count.std()

# Subset dataset based on computed boundaries
reduced_dataset = dataset[
    (dataset.raw_notes_count >= inf_bound) & (dataset.raw_notes_count <= sup_bound)
].copy()

# Compute max count to pad array to max length
max_notes_count = reduced_dataset.raw_notes_count.max()
min_notes_count = reduced_dataset.raw_notes_count.min()

# Pad notes array to max length
reduced_dataset["padded_notes"] = reduced_dataset.raw_notes.swifter.apply(pad_array)
reduced_dataset["trimed_notes"] = reduced_dataset.raw_notes.swifter.apply(lambda a: a[0:min_notes_count])

# Shuffle dataset
from sklearn.utils import shuffle

reduced_dataset = shuffle(reduced_dataset)

timesteps = max_notes_count
features = 1
output_size = len(reduced_dataset.canonical_composer.cat.categories)

train = reduced_dataset[reduced_dataset.split == "train"]
test = reduced_dataset[reduced_dataset.split == "test"]
validation = reduced_dataset[reduced_dataset.split == "validation"]
# x_train = np.vstack(train["padded_notes"]).reshape(len(train.index), -1, features)
x_train = np.vstack(train["trimed_notes"]).reshape(len(train.index), -1, features)
y_train = np.asarray(train["canonical_composer"].cat.codes)
# x_test = np.vstack(test["padded_notes"]).reshape(len(test.index), -1, features)
x_test = np.vstack(test["trimed_notes"]).reshape(len(test.index), -1, features)
y_test = np.asarray(test["canonical_composer"].cat.codes)
# x_validation = np.vstack(validation["padded_notes"]).reshape(len(validation.index), -1, features)
x_validation = np.vstack(validation["trimed_notes"]).reshape(len(validation.index), -1, features)
y_validation = np.asarray(validation["canonical_composer"].cat.codes)

print("dataset created:")
print(
    f"{x_train.shape[0]*100/(x_train.shape[0]+x_test.shape[0]):.0f}% of the data for training"
)
print(f"train: {x_train.shape}, {y_train.shape}")
print(f"test:  {x_test.shape}, {y_test.shape}")

test_model = keras.models.load_model(os.path.join(os.getcwd(), "saved_models", "the_one.hdf5"))

idx = np.random.choice(len(x_validation))
sample, sample_label = x_validation[idx], y_validation[idx]
result = tf.argmax(test_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
print(
    f"Predicted result is: {code_to_label(result.numpy())}, target result is: {code_to_label([sample_label])}"
)