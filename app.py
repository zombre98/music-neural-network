from app.features.DataMIDI import File
import json
import os


def main():
    with open('./data/maestro-v2.0.0/maestro-v2.0.0.json', 'r') as raw:
        data = json.load(raw)
        for e in data:
            filename = e['midi_filename']
            print(f'loading file {filename}')
            mid = File(os.path.join('./data/maestro-v2.0.0/' + filename))
            print(mid.features.getTempos())


if __name__ == "__main__":
    main()
