import random
import json
import numpy as np
import pandas as pd

from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam


def number_of_classes():
    with open('./data/maestro-v2.0.0/maestro-v2.0.0-authors.json', 'r') as raw:
        data = json.load(raw)
        return len(data)


def extract_features(data):
    return {'TempoMax': data.tempos.getMax(), 'TempoMin': data.tempos.getMin(),
            'NoteMax': data.notes.getMax(), 'NoteMin': data.notes.getMin(),
            'TempoRepetition': data.tempos.getRepetition(), 'NoteRepetition': data.notes.getRepetition(),
            'TemposAverageByPartsTime': data.tempos.getAverageByPartsTime(),
            'NotesAverageByPartsTime': data.notes.getAverageByPartsTime()}


TEST_DATA_PERCENTAGE = 30 / 100
NUM_CLASSES = number_of_classes()


class CNN:
    def __init__(self, data):
        self.__build_data(data)
        print({'training': len(self.training_data), 'test': len(self.test_data)})
        self.__build_model((32, 32, 3), 1)

    def __build_data(self, data):
        random.shuffle(data)
        formatted_data = [extract_features(x) for x in data]
        dataframe = pd.DataFrame.from_dict(formatted_data)
        dataframe.to_csv('data.csv')
        # self.test_data, self.training_data = np.split(formatted_data, [int(len(data) * TEST_DATA_PERCENTAGE)])

    def __build_model(self, input_shape, output_shape):
        print(f'Input shape : {input_shape}')
        print(f'Output shape : {output_shape}')
        net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                       input_shape=input_shape)
        x = net.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
        print(f'output shape : {output_layer}')
        self.model = Model(inputs=net.input, outputs=output_layer)
        # for layer in model.layers[:FREEZE_LAYERS]:
        #     layer.trainable = False
        # for layer in model.layers[FREEZE_LAYERS:]:
        #     layer.trainable = True
        self.model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
