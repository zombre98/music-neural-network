import json
import os

from app.features.DataMIDI import File


def main():
    with open('./data/maestro-v2.0.0/maestro-v2.0.0.json', 'r') as raw:
        data = json.load(raw)
        files = []
        for e in data:
            filename = e['midi_filename']
            files.append(File(os.path.join(f'./data/maestro-v2.0.0/{filename}')))


if __name__ == "__main__":
    main()
