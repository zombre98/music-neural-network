class Notes:
    def __init__(self):
        self.notesWithRepetition = []  # {'tempo': , 'repetition': , 'time': }
        self.notes = []  # {{'tempo': , 'currentTime': }
        self.time = 0

    def setTime(self, time):
        self.time = time

    def getRepetition(self):
        self.notesWithRepetition.sort(key=getRepetition, reverse=True)
        return self.notesWithRepetition

    def get(self):
        return self.notes

    def getAll(self):
        return [d['note'] for d in self.notes]

    # 0: start
    # 1: middle start
    # 2: middle
    # 3: middle end
    # 4: end
    def getAverageByPartsTime(self):
        averageMIDI = []
        j = 0
        timeSplit = self.time / 5
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
