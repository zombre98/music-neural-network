from mido import tick2second, tempo2bpm


class Tempos:
    def __init__(self):
        self.temposWithRepetition = []  # {'tempo': , 'repetition': , 'time': }
        self.tempos = []  # {{'tempo': , 'currentTime': }
        self.totalTime = 0

    def setTotalTime(self, time):
        self.totalTime = time

    def getRepetition(self):
        self.temposWithRepetition.sort(key=getRepetition, reverse=True)
        return self.temposWithRepetition

    def get(self):
        return self.tempos

    def getAll(self):
        return [d['tempo'] for d in self.tempos]

    def append(self, tempo, ticksPerBeat, time, currentTime):
        self.tempos.append({'tempo': round(tempo2bpm(tempo)), 'currentTime': currentTime})
        # fill tempo with repetition
        find = False
        for j in range(len(self.temposWithRepetition)):
            if self.temposWithRepetition[j].get('tempo') == round(tempo2bpm(tempo)):
                self.temposWithRepetition[j]['repetition'] += 1
                self.temposWithRepetition[j]['time'] += tick2second(time, ticksPerBeat, tempo)
                find = True
        if not find:
            self.temposWithRepetition.append(
                {'tempo': round(tempo2bpm(tempo)), 'repetition': 1,
                 'time': tick2second(time, ticksPerBeat, tempo)})

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
