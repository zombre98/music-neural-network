class Notes:
    def __init__(self):
        self.notesWithRepetition = []  # {'note': , 'repetition': , 'percentage': }
        self.notes = []  # {{'note': , 'currentTime': }
        self.totalTime = 0
        self.totalNotes = 0

    def setTotalTime(self, time):
        self.totalTime = time

    def getRepetition(self):
        for i in range(len(self.notesWithRepetition)):
            self.notesWithRepetition[i]['percentage'] = round((self.notesWithRepetition[i].get('repetition') / self.totalNotes) * 100, 2)
        self.notesWithRepetition.sort(key=getRepetition, reverse=True)
        return self.notesWithRepetition

    def get(self):
        return self.notes

    def getAll(self):
        return [d['note'] for d in self.notes]

    def append(self, note, currentTime):
        self.totalNotes += 1
        self.notes.append({'note': note, 'currentTime': currentTime})

        find = False
        for j in range(len(self.notesWithRepetition)):
            if self.notesWithRepetition[j].get('note') == note:
                self.notesWithRepetition[j]['repetition'] += 1
                find = True
        if not find:
            self.notesWithRepetition.append({'note': note, 'repetition': 1})

    # 0: start
    # 1: middle start
    # 2: middle
    # 3: middle end
    # 4: end
    def getAverageByPartsTime(self, nbSplit):
        averageMIDI = []
        j = 0
        timeSplit = self.totalTime / nbSplit
        currentTimeSplit = timeSplit
        average = 0
        for i in range(len(self.notes)):
            average = average + self.notes[i]['note']
            j += 1
            if self.notes[i].get('currentTime') >= currentTimeSplit or i == len(self.notes) - 1:
                averageMIDI.append(round(average / j, 2))
                currentTimeSplit += timeSplit
                j = 0
                average = 0
        return averageMIDI

    def getMax(self):
        return max([d['note'] for d in self.notes])

    def getMin(self):
        return min([d['note'] for d in self.notes])


def getRepetition(tempos):
    return tempos.get('repetition')
