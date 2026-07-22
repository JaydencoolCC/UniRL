import json
import logging

import pytest

from unirl.data.sft import SupervisedDataSource


def _write_manifest(path, prefix, count=6):
    with path.open("w") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "sample_id": f"{prefix}-{index}",
                        "prompt": f"{prefix} prompt {index}",
                        "response": f"{prefix} response {index}",
                    }
                )
                + "\n"
            )


def test_state_dict_fingerprints_manifest_and_resumes_exactly(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, "a")
    source = SupervisedDataSource(str(manifest), seed=7)
    source.get_samples(4)
    state = source.state_dict()
    expected = source.get_samples(5)

    assert state["manifest_fingerprint"].startswith("sha256:")
    assert state["num_records"] == 6

    resumed = SupervisedDataSource(str(manifest), seed=7)
    resumed.load_state_dict(state)
    assert resumed.get_samples(5) == expected


def test_load_state_dict_rejects_same_size_changed_manifest(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, "a")
    state = SupervisedDataSource(str(manifest)).state_dict()

    _write_manifest(manifest, "b")
    changed = SupervisedDataSource(str(manifest))
    with pytest.raises(ValueError, match="manifest fingerprint mismatch"):
        changed.load_state_dict(state)


def test_load_state_dict_accepts_legacy_state_with_warning(tmp_path, caplog):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, "a")
    source = SupervisedDataSource(str(manifest), seed=3)

    with caplog.at_level(logging.WARNING):
        source.load_state_dict({"epoch": 0, "position": 2, "seed": 3})

    assert "legacy checkpoint has no manifest fingerprint" in caplog.text
    assert source.state_dict()["position"] == 2


def test_eval_is_empty_without_explicit_manifest(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, "train")
    source = SupervisedDataSource(str(manifest))

    assert not source.has_eval_data
    assert list(source.iter_eval_batches(2)) == []


def test_explicit_train_manifest_can_still_be_used_for_eval(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, "train", count=5)
    source = SupervisedDataSource(str(manifest), eval_manifest_path=str(manifest))

    batches = list(source.iter_eval_batches(2))

    assert source.has_eval_data
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [row["sample_id"] for batch in batches for row in batch] == [
        f"train-{index}" for index in range(5)
    ]


def test_peek_keeps_checkpoint_cursor_until_committed(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, "train", count=6)
    source = SupervisedDataSource(str(manifest), seed=11)
    reference = SupervisedDataSource(str(manifest), seed=11)
    first = source.get_samples(2)
    reference.get_samples(2)
    state_before = source.state_dict()
    expected = reference.get_samples(3)

    peeked = source.peek_samples(3)

    assert peeked == expected
    assert source.state_dict() == state_before
    with pytest.raises(RuntimeError, match="commit the pending"):
        source.get_samples(1)
    assert source.commit_peeked_samples() == peeked
    assert source.state_dict() == reference.state_dict()
    assert first != peeked
