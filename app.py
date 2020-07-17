import json
import os

from app.features.DataMIDI import File
from app.cnn.cnn import CNN


def main():
    with open('./data/maestro-v2.0.0/maestro-v2.0.0.json', 'r') as raw:
        data = json.load(raw)
        files = []
        # for e in data:
        #     filename = e['midi_filename']
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'))
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi'))
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_08_Track08_wav.midi'))
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'))
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi'))
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_08_Track08_wav.midi'))
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'))
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi'))
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_08_Track08_wav.midi'))
        files.append(File('/Users/zombre/Documents/Kent/music-neural-network/data/maestro-v2.0.0/2004/MIDI'
                          '-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_08_Track08_wav.midi'))
        cnn = CNN(files)



if __name__ == "__main__":
    main()
