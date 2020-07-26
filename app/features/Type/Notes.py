class Notes:
    def __init__(self):
        self.notesWithRepetition = []  # {'note': , 'repetition': , 'time': }
        self.notes = []  # {{'note': , 'currentTime': }
        self.totalTime = 0

    def setTotalTime(self, time):
        self.totalTime = time

    def getRepetition(self):
        self.notesWithRepetition.sort(key=getRepetition, reverse=True)
        return self.notesWithRepetition

    def get(self):
        return self.notes

    def getAll(self):
        return [d['note'] for d in self.notes]

    def append(self, note, currentTime, deltaTime):
        self.notes.append({'note': note, 'currentTime': currentTime, 'deltaTime': deltaTime})

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
    def getAverageByPartsTime(self):
        averageMIDI = []
        j = 0
        timeSplit = self.totalTime / 5
        currentTimeSplit = timeSplit
        average = 0
        for i in range(len(self.notes)):
            average = average + self.notes[i]['note']
            j = j + 1
            if self.notes[i].get('currentTime') >= currentTimeSplit or currentTimeSplit == (timeSplit * 5):
                averageMIDI.append(round(average / j, 2))
                currentTimeSplit = currentTimeSplit + timeSplit
                j = 0
                average = 0
        return averageMIDI

    def getMax(self):
        return max([d['note'] for d in self.notes])

    def getMin(self):
        return min([d['note'] for d in self.notes])


def getRepetition(tempos):
    return tempos.get('repetition')
