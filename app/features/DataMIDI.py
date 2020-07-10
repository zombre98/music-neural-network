from mido import MidiFile, tempo2bpm, tick2second
from app.features.FeaturesMIDI import FeaturesMIDI


class File:
    def __init__(self, file_name):
        self.mid = MidiFile(file_name)
        self.features = FeaturesMIDI()
        self.parseData()

    def parseData(self):
        tempoTemp = 0
        for i, track in enumerate(self.mid.tracks):
            time = 0
            # print('Track {}: {}'.format(i, track.name))
            for msg in track:
                time = tick2second(msg.time, self.mid.ticks_per_beat, tempoTemp) + time
                if msg.type == 'set_tempo':
                    tempoTemp = msg.tempo
                    # fill tempo
                    self.features.tempos.append({'tempo': round(tempo2bpm(msg.tempo)), 'currentTime': time})
                    # fill tempo with repetition
                    find = False
                    for j in range(len(self.features.temposWithRepetition)):
                        if self.features.temposWithRepetition[j].get('tempo') == round(tempo2bpm(msg.tempo)):
                            self.features.temposWithRepetition[j]['repetition'] = self.features.temposWithRepetition[j][
                                                                                      'repetition'] + 1
                            self.features.temposWithRepetition[j]['time'] = tick2second(msg.time, self.mid.ticks_per_beat, msg.tempo) + self.features.temposWithRepetition[j]['time']
                            find = True
                    if not find:
                        self.features.temposWithRepetition.append(
                            {'tempo': round(tempo2bpm(msg.tempo)), 'repetition': 1, 'time': tick2second(msg.time, self.mid.ticks_per_beat, msg.tempo)})
            if i == 0:
                self.features.time = time
