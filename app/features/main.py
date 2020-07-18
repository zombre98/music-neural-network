import sys
from app.features.MidiData import MidiData


def main(argv):
    mid = MidiData(argv[0])

    print('=================== TEMPOS ===================')

    print('====== REPETITION ======')
    print(mid.tempos.getRepetition())

    print('======== TEMPOS WITH TIME ========')
    print(mid.tempos.get())

    print('==== SIMPLY TEMPOS ====')
    print(mid.tempos.getAll())

    print('= AVERAGE TEMPOS TIME =')
    print(mid.tempos.getAverageByPartsTime(1))

    print('====== MAX TEMPO =======')
    print(mid.tempos.getMax())

    print('====== MIN TEMPO =======')
    print(mid.tempos.getMin())

    print('=================== NOTES ===================')

    print('====== REPETITION ======')
    print(mid.notes.getRepetition())

    print('======== NOTES WITH TIME ========')
    print(mid.notes.get())

    print('==== SIMPLY NOTES ====')
    print(mid.notes.getAll())

    print('= AVERAGE NOTES TIME =')
    print(mid.notes.getAverageByPartsTime(10))

    print('====== MAX NOTE =======')
    print(mid.notes.getMax())

    print('====== MIN NOTE =======')
    print(mid.notes.getMin())


if __name__ == "__main__":
    if sys.argv[1:]:
        main(sys.argv[1:])
    else:
        print('Error argument')
