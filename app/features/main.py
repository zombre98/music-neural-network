import sys
from app.features.MidiData import MidiData


def main(argv):
    mid = MidiData(argv[0])

    print('=========== TEMPOS =============')

    print('======== DATA ========')
    print(mid.tempos.get())

    print('===== REPETITION =====')
    print(mid.tempos.getRepetition())

    print('== AVERAGE BY PART ==')
    print(mid.tempos.getAverageByPartsTime(2))

    print('======== MAX =========')
    print(mid.tempos.getMax())

    print('======== MIN =========')
    print(mid.tempos.getMin())

    print('=========== NOTES =============')

    print('======== DATA ========')
    print(mid.notes.get())

    print('===== REPETITION =====')
    print(mid.notes.getRepetition())

    print('== AVERAGE BY PART ==')
    print(mid.notes.getAverageByPartsTime(2))

    print('======== MAX =========')
    print(mid.notes.getMax())

    print('======== MIN =========')
    print(mid.notes.getMin())


if __name__ == "__main__":
    if sys.argv[1:]:
        main(sys.argv[1:])
    else:
        print('Error argument')
