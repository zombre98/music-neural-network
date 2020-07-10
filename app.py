from app.features.DataMIDI import File
import time
import json
import os


def main():
    with open('./data/maestro-v2.0.0/maestro-v2.0.0.json', 'r') as raw:
        data = json.load(raw)
        start_time = time.time()
        files = []
        for e in data:
            filename = e['midi_filename']
            files.append(File(os.path.join(f'./data/maestro-v2.0.0/{filename}')))
        current_time = time.time()
        print(f'Elapsed time: {current_time - start_time}')


if __name__ == "__main__":
    main()
