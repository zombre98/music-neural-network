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
current_time = time.time()
print(f'Elapsed time: {(current_time - start_time) / 60} minute(s)')


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
            keras.layers.Dense(2, activation="relu", name="input_layer", input_shape=input_shape),
            keras.layers.Flatten(name='Flatten'),
            keras.layers.Dense(output_shape, name="output_layer"),
        ]
    )
    model_to_build.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model_to_build.summary()
    return model_to_build


##
def format_data(data_to_format):
    x_data = np.vstack(data_to_format.midi_data.map(lambda x: np.array([x.notes.getMax(), x.notes.getMin()])))
    y_data = np.asarray(data_to_format['canonical_composer'].cat.codes)

    return x_data, y_data


TEST_DATA_PERCENTAGE = 20 / 100
NUM_CLASSES = number_of_classes()

##
x_data, y_data = format_data(dataset)

##
model = build_model(x_data[0].shape, NUM_CLASSES)

kfold = KFold(n_splits=5)

##
fold_no = 1
for train_index, test_index in kfold.split(x_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    checkpoint = keras.callbacks.ModelCheckpoint(f'./models/models_{fold_no}.h5',
                                                 monitor='val_accuracy', verbose=0,
                                                 save_best_only=True, mode='max')
    model.fit(x_train, y_train, steps_per_epoch=10, epochs=15, callbacks=[checkpoint],
              validation_data=(x_test, y_test), use_multiprocessing=True, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, steps=10, use_multiprocessing=True, verbose=0)
    print(f'Model n°{fold_no} has achieved {accuracy}% of accuracy with {loss} loss')
    fold_no += 1

##
idx = np.random.choice(len(x_data))
sample, sample_label = x_data[idx], y_data[idx]

test_model = build_model(x_data[0].shape, NUM_CLASSES)
test_model.set_weights(model.get_weights())
result = tensorflow.argmax(test_model.predict_on_batch(tensorflow.expand_dims(sample, 0)), axis=1)
print(f'Predicted result is: {code_to_label(result)}, target result is: {code_to_label(sample_label)}, idx: {idx}')
