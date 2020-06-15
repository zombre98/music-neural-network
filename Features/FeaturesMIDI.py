import numpy as np


class FeaturesMIDI:
    def __init__(self):
        self.temposWithRepetition = []
        self.tempos = []

    def getRepetitionOfTempos(self):
        self.temposWithRepetition.sort(key=getRepetition, reverse=True)
        return self.temposWithRepetition

    def getTempos(self):
        return self.tempos

    def getTemposAverage(self):
        return round(sum(self.tempos) / len(self.tempos), 2)

    # 0: start
    # 1: middle start
    # 2: middle
    # 3: middle end
    # 4: end
    def getTemposAverageByParts(self):
        averageMIDI = []
        temposSplit5 = np.array_split(self.tempos, 5)
        for i in range(len(temposSplit5)):
            average = 0
            for j in range(len(temposSplit5[i])):
                average = average + temposSplit5[i][j]
            averageMIDI.append(round(average / len(temposSplit5[i]), 2))
        return averageMIDI

    def getMaxTempo(self):
        return max(self.tempos)

    def getMinTempo(self):
        return min(self.tempos)


def getRepetition(tempos):
    return tempos.get('repetition')
