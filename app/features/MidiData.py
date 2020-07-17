from mido import MidiFile, tempo2bpm, tick2second, merge_tracks
from app.features.Type.Notes import Notes
from app.features.Type.Tempos import Tempos

DEFAULT_TEMPO = 500000

class MidiData:
    def __init__(self, file_name):
        self.tempos = Tempos()
        self.notes = Notes()

        self.parseData(file_name)

    def parseData(self, file_name):
        mid = MidiFile(file_name)
        currentTime = 0
        tempo = DEFAULT_TEMPO
        for msg in merge_tracks(mid.tracks):
            if msg.time > 0:
                delta = tick2second(msg.time, mid.ticks_per_beat, tempo)
            else:
                delta = 0
            currentTime += delta

            if msg.type == 'set_tempo':
                tempo = msg.tempo
                # fill tempo
                self.tempos.append(msg.tempo, mid.ticks_per_beat, msg.time, currentTime)
            elif msg.type == 'note_on':
                self.notes.append(msg.note, currentTime)
        self.notes.setTotalTime(currentTime)
        self.tempos.setTotalTime(currentTime)
