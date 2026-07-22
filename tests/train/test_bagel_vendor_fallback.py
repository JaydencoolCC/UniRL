import importlib.util
import sys

import unirl.models.bagel.vendor  # noqa: F401


def test_flash_attention_fallback_has_import_spec():
    module = sys.modules["flash_attn"]

    assert module.__spec__ is not None
    assert importlib.util.find_spec("flash_attn") is not None
