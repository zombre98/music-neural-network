from mido import MidiFile, tempo2bpm
import FeaturesMIDI


class File:
    def __init__(self, file_name):
        self.mid = MidiFile(file_name)
        self.features = FeaturesMIDI.FeaturesMIDI()
        self.parseData()

    def parseData(self):
        for i, track in enumerate(self.mid.tracks):
            # print('Track {}: {}'.format(i, track.name))
            for msg in track:
                if msg.type == 'set_tempo':
                    # fill tempo
                    self.features.tempos.append(round(tempo2bpm(msg.tempo)))
                    # fill tempo with repetition
                    find = False
                    for j in range(len(self.features.temposWithRepetition)):
                        if self.features.temposWithRepetition[j].get('tempo') == round(tempo2bpm(msg.tempo)):
                            self.features.temposWithRepetition[j]['repetition'] = self.features.temposWithRepetition[j][
                                                                                      'repetition'] + 1
                            find = True
                    if not find:
                        self.features.temposWithRepetition.append(
                            {'tempo': round(tempo2bpm(msg.tempo)), 'repetition': 1})
                    else:
                        find = False