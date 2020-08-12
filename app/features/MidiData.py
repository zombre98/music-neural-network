from mido import MidiFile, tempo2bpm, tick2second, merge_tracks
from app.features.TypeData import TypeData

DEFAULT_TEMPO = 500000

class MidiData:
    def __init__(self, file_name):
        self.tempos = TypeData("tempo")
        self.notes = TypeData("note")

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
                self.tempos.append(tempo2bpm(msg.tempo), currentTime)
            elif msg.type == 'note_on':
                self.notes.append(msg.note, currentTime)
        self.notes.createDataFrame()
        self.tempos.createDataFrame()
