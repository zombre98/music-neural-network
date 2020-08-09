##
import os
import time

import keras
import pandas as pd
import swifter
import numpy as np
import json

from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import KFold
import tensorflow

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


def number_of_classes():
    with open('./data/maestro-v2.0.0/maestro-v2.0.0-authors.json', 'r') as raw:
        data = json.load(raw)
        return len(data)


def get_midi_data(filename):
    return MidiData(os.path.join("./data/maestro-v2.0.0/", filename))

##
data = pd.read_json('./data/maestro-v2.0.0/maestro-v2.0.0.json')
dataset = data.astype(DTYPE)
start_time = time.time()
##
dataset['midi_data'] = dataset['midi_filename'].swifter.progress_bar().allow_dask_on_strings().apply(get_midi_data)
##
import pickle

with open('./dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
##
current_time = time.time()
print(f'Elapsed time: {(current_time - start_time) / 60} minute(s)')


##
def get_notes(midi_data):
    return np.asarray([n["note"] for n in midi_data.notes.notes])


dataset['raw_notes'] = dataset.midi_data.swifter.apply(get_notes)
dataset['raw_notes_count'] = dataset.raw_notes.swifter.apply(lambda a: len(a))


def pad_array(a):
    return np.pad(a, (0, max_notes_count - a.shape[0]), mode="constant")


max_notes_count = dataset.raw_notes_count.max()
min_notes_count = dataset.raw_notes_count.min()

# Pad notes array to max length
dataset["padded_notes"] = dataset.raw_notes.swifter.apply(pad_array)


##
def code_to_label(code):
    return dataset.canonical_composer.cat.categories[code]


def extract_features(features):
    midi_data = features['midi_data']
    return {'Composer': features['canonical_composer'],
            'TempoMax': midi_data.tempos.getMax(), 'TempoMin': midi_data.tempos.getMin(),
            'NoteMax': midi_data.notes.getMax(), 'NoteMin': midi_data.notes.getMin(),
            'TempoRepetition': midi_data.tempos.getRepetition(), 'NoteRepetition': midi_data.notes.getRepetition(),
            'TemposAverageByPartsTime': midi_data.tempos.getAverageByPartsTime(),
            'NotesAverageByPartsTime': midi_data.notes.getAverageByPartsTime()}


##
def build_model(input_shape, output_shape):
    print(f'Input shape : {input_shape}')
    print(f'Output shape : {output_shape}')
    model_to_build = keras.Sequential(
        [
            keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(name='flatten_layer'),
            # keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(output_shape),
        ]
    )
    model_to_build.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
    model_to_build.summary()
    return model_to_build


##
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


NUM_CLASSES = number_of_classes()

##
x_train, y_train, x_test, y_test, x_validation, y_validation = format_data(dataset)

##
model = build_model(x_train[0].shape, NUM_CLASSES)

##
keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

##
early_stopping = keras.callbacks.EarlyStopping(patience=10)
checkpoint = keras.callbacks.ModelCheckpoint(f'./models/model.h5',
                                             monitor='val_accuracy', verbose=2,
                                             save_best_only=True, mode='max')
history = model.fit(x_train, y_train, batch_size=100, epochs=16,
                    callbacks=[early_stopping, checkpoint],
                    validation_data=(x_test, y_test), use_multiprocessing=True, verbose=2)
loss, accuracy = model.evaluate(x_test, y_test, steps=10, use_multiprocessing=True, verbose=2)
print(f'Model has achieved {accuracy}% of accuracy with {loss} loss')

##
pd.DataFrame(history.history)[["loss", "val_loss"]].plot().set(xlabel="Epoch", ylabel="Loss")
pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot().set(xlabel="Epoch", ylabel="Accuracy")

##
idx = np.random.choice(len(x_test))
sample, sample_label = x_test[idx], y_test[idx]

test_model = build_model(x_test[0].shape, NUM_CLASSES)
test_model.set_weights(model.get_weights())
result = tensorflow.argmax(test_model.predict_on_batch(tensorflow.expand_dims(sample, 0)), axis=1)
print(f'Predicted result is: {code_to_label(result)}, target result is: {code_to_label(sample_label)}, idx: {idx}')
