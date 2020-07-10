from mido import MidiFile, tempo2bpm, tick2second, merge_tracks
from Features.Type.Notes import Notes
from Features.Type.Tempos import Tempos


class MidiData:
    def __init__(self, file_name):
        self.mid = MidiFile(file_name)
        self.tempos = Tempos()
        self.notes = Notes()

        self.parseData()

    def parseData(self):
        print("Data recovery...")

        self.notes.setTime(self.mid.length)
        self.tempos.setTime(self.mid.length)
        currentTime = 0
        tempoTemp = 0
        for msg in merge_tracks(self.mid.tracks):
            currentTime = tick2second(msg.time, self.mid.ticks_per_beat, tempoTemp) + currentTime
            if msg.type == 'set_tempo':
                tempoTemp = msg.tempo
                # fill tempo
                self.tempos.tempos.append({'tempo': round(tempo2bpm(msg.tempo)), 'currentTime': currentTime})
                # fill tempo with repetition
                find = False
                for j in range(len(self.tempos.temposWithRepetition)):
                    if self.tempos.temposWithRepetition[j].get('tempo') == round(tempo2bpm(msg.tempo)):
                        self.tempos.temposWithRepetition[j]['repetition'] = self.tempos.temposWithRepetition[j][
                                                                                  'repetition'] + 1
                        self.tempos.temposWithRepetition[j]['time'] = tick2second(msg.time, self.mid.ticks_per_beat, msg.tempo) + self.tempos.temposWithRepetition[j]['time']
                        find = True
                if not find:
                    self.tempos.temposWithRepetition.append(
                        {'tempo': round(tempo2bpm(msg.tempo)), 'repetition': 1, 'time': tick2second(msg.time, self.mid.ticks_per_beat, msg.tempo)})
            elif msg.type == 'note_on':
                self.notes.notes.append({'note': msg.note, 'currentTime': currentTime})

                find = False
                for j in range(len(self.notes.notesWithRepetition)):
                    if self.notes.notesWithRepetition[j].get('note') == msg.note:
                        self.notes.notesWithRepetition[j]['repetition'] = self.notes.notesWithRepetition[j][
                                                                                  'repetition'] + 1
                        find = True
                if not find:
                    self.notes.notesWithRepetition.append({'note': msg.note, 'repetition': 1})
