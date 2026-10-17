import itertools

import pytest

from behaviours.sandbagging.prompts import CONDITION_SENTENCES, PARAPHRASES, system_prompt


@pytest.mark.parametrize("paraphrase", range(len(PARAPHRASES)))
def test_conditions_differ_only_in_their_sentence(paraphrase: int) -> None:
    stripped = {
        system_prompt(c, paraphrase).replace(sentence, "")
        for c, sentence in CONDITION_SENTENCES.items()
    }
    assert len(stripped) == 1


def test_condition_sentences_matched_in_length() -> None:
    words = [len(s.split()) for s in CONDITION_SENTENCES.values()]
    assert max(words) - min(words) <= 2


@pytest.mark.parametrize(
    ("condition", "paraphrase"),
    list(itertools.product(CONDITION_SENTENCES, range(len(PARAPHRASES)))),
)
def test_every_prompt_asks_for_answer_line(condition: str, paraphrase: int) -> None:
    assert "ANSWER: <integer>" in system_prompt(condition, paraphrase)


def test_paraphrases_distinct() -> None:
    assert len(set(PARAPHRASES)) == len(PARAPHRASES)
