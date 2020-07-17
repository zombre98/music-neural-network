import json
import os
import time
import pandas as pd
import swifter

from app.features.MidiData import MidiData
from app.cnn.cnn import CNN


def get_midi_data(filename):
    return MidiData(os.path.join("./data/maestro-v2.0.0/", filename))


def main():
    data = pd.read_json('./data/maestro-v2.0.0/maestro-v2.0.0.json')
    start_time = time.time()
    files = data['midi_filename'].swifter.progress_bar().allow_dask_on_strings().apply(get_midi_data)
    current_time = time.time()
    print(f'Elapsed time: {current_time - start_time}')
    cnn = CNN(files)


if __name__ == "__main__":
    main()
