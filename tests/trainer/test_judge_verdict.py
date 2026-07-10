"""CPU unit test for LLM-judge verdict parsing (``_parse_verdict``).

Guards the negative-first parse against the "incorrect" ⊃ "correct" substring
trap and "not correct" / "wrong" phrasings that the prior substring test
(``"correct" in content and "incorrect" not in content``) misread as correct.
Pure function — no GPU, no HTTP, no model.
"""

import pytest

from unirl.reward.local.llm_judge import _parse_verdict


@pytest.mark.parametrize(
    "reply, expected",
    [
        # plain single-word verdicts (the prompt asks for one word)
        ("correct", 1.0),
        ("incorrect", 0.0),
        ("Correct", 1.0),
        ("INCORRECT", 0.0),
        # the substring trap the old code failed: "not correct" ⊃ "correct"
        ("not correct", 0.0),
        ("The predicted answer is not correct.", 0.0),
        ("isn't correct", 0.0),
        ("wrong", 0.0),
        ("This is wrong.", 0.0),
        ("not equivalent", 0.0),
        # positive phrasings
        ("The answer is correct.", 1.0),
        ("equivalent", 1.0),
        ("yes", 1.0),
        ("Yes, they are equivalent.", 1.0),
        # verbose chain-of-thought endings
        ("The reference is 42 and the prediction is 42, therefore correct", 1.0),
        ("The prediction differs from the reference, so incorrect", 0.0),
        # ambiguous / empty -> under-reward (judge-failure convention)
        ("", 0.0),
        ("   ", 0.0),
        ("unclear", 0.0),
        ("maybe", 0.0),
    ],
)
def test_parse_verdict(reply, expected):
    assert _parse_verdict(reply) == expected
