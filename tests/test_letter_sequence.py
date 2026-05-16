import random
from src.backend.core.letter_sequence import LetterSequence


def test_sequential_progression():
    seq = LetterSequence(mode="sequential", include_dynamic=False)
    first = seq.get_next_letter()
    second = seq.get_next_letter(first)
    assert first != second


def test_random_avoids_same():
    random.seed(0)
    seq = LetterSequence(mode="random", include_dynamic=False)
    cur = seq.get_next_letter()
    nxt = seq.get_next_letter(cur)
    assert nxt != cur or len(seq.available_letters) == 1
