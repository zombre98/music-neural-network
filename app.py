from mido import MidiFile
import json
import os

with open('./data/maestro-v2.0.0/maestro-v2.0.0.json', 'r') as raw:
    data = json.load(raw)
    for e in data:
        filename = e['midi_filename']
        print(f'loading file {filename}')
        mid = MidiFile(os.path.join('./data/maestro-v2.0.0/' + filename))
        print(len(mid.tracks))

# for i, track in enumerate(mid.tracks):
    # print('Track {}: {} size {}'.format(i, track.name, len(track)))
