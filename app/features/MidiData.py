from mido import MidiFile, tempo2bpm, tick2second, merge_tracks
from app.features.Type.Notes import Notes
from app.features.Type.Tempos import Tempos


class MidiData:
    def __init__(self, file_name):
        self.mid = MidiFile(file_name)
        self.tempos = Tempos()
        self.notes = Notes()

        self.parseData()

    def parseData(self):
        midi_length = self.mid.length
        self.notes.setTotalTime(midi_length)
        self.tempos.setTotalTime(midi_length)
        currentTime = 0
        tempoTemp = 0
        for msg in merge_tracks(self.mid.tracks):
            currentTime += tick2second(msg.time, self.mid.ticks_per_beat, tempoTemp)
            if msg.type == 'set_tempo':
                tempoTemp = msg.tempo
                # fill tempo
                self.tempos.append(msg.tempo, self.mid.ticks_per_beat, msg.time, currentTime)
            elif msg.type == 'note_on':
                self.notes.append(msg.note, currentTime)
