class Tempos:
    def __init__(self):
        self.temposWithRepetition = []  # {'tempo': , 'repetition': , 'time': }
        self.tempos = []  # {{'tempo': , 'currentTime': }
        self.time = 0

    def setTime(self, time):
        self.time = time

    def getRepetition(self):
        self.temposWithRepetition.sort(key=getRepetition, reverse=True)
        return self.temposWithRepetition

    def get(self):
        return self.tempos

    def getAll(self):
        return [d['tempo'] for d in self.tempos]

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
        for i in range(len(self.tempos)):
            average = average + self.tempos[i]['tempo']
            j = j + 1
            if self.tempos[i].get('currentTime') >= currentTimeSplit or currentTimeSplit == (timeSplit * 5):
                averageMIDI.append(round(average / j, 2))
                currentTimeSplit = currentTimeSplit + timeSplit
                j = 0
                average = 0
        return averageMIDI

    def getMax(self):
        return max([d['tempo'] for d in self.tempos])

    def getMin(self):
        return min([d['tempo'] for d in self.tempos])


def getRepetition(tempos):
    return tempos.get('repetition')
