import sys
import DataMIDI


def main(argv):
    mid = DataMIDI.File(argv[0])

    print('====== REPETITION ======')
    print(mid.features.getRepetitionOfTempos())

    print('======== TEMPOS ========')
    print(mid.features.getTempos())

    print('==== AVERAGE TEMPOS ====')
    print(mid.features.getTemposAverage())

    # print('== AVERAGE TEMPOS PART ==')
    # print(mid.features.getTemposAverageByParts())

    print('= AVERAGE TEMPOS TIME =')
    print(mid.features.getTemposAverageByPartsTime())

    print('====== MAX TEMPO =======')
    print(mid.features.getMaxTempo())

    print('====== MIN TEMPO =======')
    print(mid.features.getMinTempo())


if __name__ == "__main__":
    if sys.argv[1:]:
        main(sys.argv[1:])
    else:
        print('Error argument')
