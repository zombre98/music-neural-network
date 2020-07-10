import numpy as np


class FeaturesMIDI:
    def __init__(self):
        self.temposWithRepetition = []
        self.tempos = []
        self.time = 0

    def getRepetitionOfTempos(self):
        self.temposWithRepetition.sort(key=getRepetition, reverse=True)
        return self.temposWithRepetition

    def getTempos(self):
        return self.tempos

    def getTemposAverage(self):
        return [d['tempo'] for d in self.tempos]

    # 0: start
    # 1: middle start
    # 2: middle
    # 3: middle end
    # 4: end
    def getTemposAverageByParts(self):
        averageMIDI = []
        temposSplit5 = np.array_split([d['tempo'] for d in self.tempos], 5)
        for i in range(len(temposSplit5)):
            average = 0
            for j in range(len(temposSplit5[i])):
                average = average + temposSplit5[i][j]
            averageMIDI.append(round(average / len(temposSplit5[i]), 2))
        return averageMIDI

    # 0: start
    # 1: middle start
    # 2: middle
    # 3: middle end
    # 4: end
    def getTemposAverageByPartsTime(self):
        averageMIDI = []
        j = 0
        timeSplit = self.time / 5
        currentTimeSplit = timeSplit
        average = 0
        for i in range (len(self.tempos)):
            average = average + self.tempos[i]['tempo']
            j = j + 1
            if self.tempos[i].get('currentTime') >= currentTimeSplit or currentTimeSplit == (timeSplit * 5):
                averageMIDI.append(round(average / j, 2))
                currentTimeSplit = currentTimeSplit + timeSplit
                j = 0
                average = 0
        return averageMIDI

    def getMaxTempo(self):
        return max([d['tempo'] for d in self.tempos])

    def getMinTempo(self):
        return min([d['tempo'] for d in self.tempos])


def getRepetition(tempos):
    return tempos.get('repetition')
