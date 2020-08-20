import pandas as pd


class TypeData:
    def __init__(self, nameType):
        self.dataFrame = None
        self.data = []
        self.nameType = nameType

    def getRepetition(self):
        df = pd.DataFrame({'percentage': self.dataFrame[self.nameType].value_counts(normalize=True) * 100,
                           'count': self.dataFrame.pivot_table(index=[self.nameType], aggfunc='size')})
        return df

    def get(self):
        return self.dataFrame

    def append(self, note, currentTime):
        self.data.append([note, currentTime])

    def getAverageByPartsTime(self, nbSplit):
        arrayAverage = []
        timeToSplit = self.dataFrame['currentTime'].iloc[-1] / nbSplit
        i = 0

        while i < nbSplit:
            arrayAverage.append(round(pd.DataFrame(self.dataFrame[(self.dataFrame['currentTime'] < timeToSplit * (i + 1)) & (self.dataFrame['currentTime'] >= timeToSplit * i)])[self.nameType].mean(), 2))
            i += 1

        return arrayAverage

    def getMax(self):
        return self.dataFrame[self.nameType].max()

    def getMin(self):
        return self.dataFrame[self.nameType].min()

    def createDataFrame(self):
        self.dataFrame = pd.DataFrame(self.data, columns=[self.nameType, 'currentTime'])
