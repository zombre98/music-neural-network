import sys
from Features import MidiData


def main(argv):
    mid = MidiData.MidiData(argv[0])

    print('=================== TEMPOS ===================')

    print('====== REPETITION ======')
    print(mid.tempos.getRepetition())

    print('======== TEMPOS WITH TIME ========')
    print(mid.tempos.get())

    print('==== SIMPLY TEMPOS ====')
    print(mid.tempos.getAll())

    print('= AVERAGE TEMPOS TIME =')
    print(mid.tempos.getAverageByPartsTime())

    print('====== MAX TEMPO =======')
    print(mid.tempos.getMax())

    print('====== MIN TEMPO =======')
    print(mid.tempos.getMin())

    print('=================== NOTES ===================')

    print('======== NOTES WITH TIME ========')
    print(mid.notes.get())

    print('==== SIMPLY NOTES ====')
    print(mid.notes.getAll())

    print('====== REPETITION ======')
    print(mid.notes.getRepetition())

    print('= AVERAGE NOTES TIME =')
    print(mid.notes.getAverageByPartsTime())

    print('====== MAX NOTE =======')
    print(mid.notes.getMax())

    print('====== MIN NOTE =======')
    print(mid.notes.getMin())


if __name__ == "__main__":
    if sys.argv[1:]:
        main(sys.argv[1:])
    else:
        print('Error argument')
