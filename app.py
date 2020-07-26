import json
import os
import time
import pandas as pd
import swifter

from app.features.MidiData import MidiData
from app.cnn.cnn import CNN

DTYPE = {
    'canonical_composer': 'category',
    'canonical_title': 'object',
    'split': 'category',
    'year': 'int64',
    'midi_filename': 'object',
    'audio_filename': 'object',
    'duration': 'float64'
}


def get_midi_data(filename):
    return MidiData(os.path.join("./data/maestro-v2.0.0/", filename))


def main():
    data = pd.read_json('./data/maestro-v2.0.0/maestro-v2.0.0.json')
    dataset = data.astype(DTYPE)
    start_time = time.time()
    dataset['midi_data'] = dataset['midi_filename'].swifter.progress_bar().allow_dask_on_strings().apply(get_midi_data)
    current_time = time.time()
    print(f'Elapsed time: {(current_time - start_time) / 60} minute(s)')
    cnn = CNN(dataset)


if __name__ == "__main__":
    main()
