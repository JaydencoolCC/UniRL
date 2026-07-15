"""Supervised data layer: manifest parsing, legacy rejection, epoch cursor resume."""

from __future__ import annotations

import json

import pytest

from unirl.data.sft import SupervisedDataset, SupervisedDataSource, normalize_supervised_example


def _write_manifest(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return str(path)


def test_normalize_blesses_response_and_target_media(tmp_path):
    record = normalize_supervised_example(
        {
            "prompt": "a cat",
            "response": "meow",
            "media": [{"modality": "image", "role": "target", "uri": "img/x.png"}],
            "extra": 1,
        },
        default_sample_id="s0",
        base_dir=str(tmp_path),
    )
    assert record["response"] == "meow"
    assert record["media_refs"][0].role == "target"
    assert record["media_refs"][0].uri.startswith(str(tmp_path))  # relative → manifest-dir resolved
    assert record["metadata"] == {"extra": 1}


def test_normalize_rejects_legacy_embedding_fields():
    with pytest.raises(ValueError, match="legacy embedding"):
        normalize_supervised_example({"prompt": "p", "prompt_embeds": []}, default_sample_id="s0")


def test_dataset_parses_jsonl_and_rejects_bad_rows(tmp_path):
    path = _write_manifest(tmp_path / "m.jsonl", [{"prompt": "p", "response": "r"}, {"caption": "c"}])
    ds = SupervisedDataset(path)
    assert len(ds) == 2
    assert ds[1]["prompt"] == "c"  # caption alias
    with pytest.raises(ValueError, match="non-empty 'prompt'"):
        SupervisedDataset(_write_manifest(tmp_path / "bad.jsonl", [{"response": "r"}]))


def test_epoch_cursor_resume_is_exact(tmp_path):
    rows = [{"prompt": f"p{i}", "response": "r"} for i in range(10)]
    path = _write_manifest(tmp_path / "m.jsonl", rows)

    src = SupervisedDataSource(path, seed=7)
    consumed = [src.get_samples(4) for _ in range(3)]  # crosses the epoch boundary (10 rows / 4)
    state = src.state_dict()
    next_batches = [src.get_samples(4) for _ in range(3)]

    resumed = SupervisedDataSource(path, seed=7)
    resumed.load_state_dict(state)
    replayed = [resumed.get_samples(4) for _ in range(3)]
    assert [[r["prompt"] for r in b] for b in replayed] == [[r["prompt"] for r in b] for b in next_batches]
    assert consumed  # silence unused warning


def test_epoch_property_tracks_position(tmp_path):
    path = _write_manifest(tmp_path / "m.jsonl", [{"prompt": f"p{i}", "response": "r"} for i in range(8)])
    src = SupervisedDataSource(path, seed=0)
    src.get_samples(4)
    assert src.epoch == pytest.approx(0.5)
    src.get_samples(8)  # wraps into epoch 1
    assert src.epoch == pytest.approx(1.0 + 4 / 8)


def test_eval_iteration_is_deterministic_and_capped(tmp_path):
    train = _write_manifest(tmp_path / "t.jsonl", [{"prompt": f"t{i}", "response": "r"} for i in range(4)])
    evalp = _write_manifest(tmp_path / "e.jsonl", [{"prompt": f"e{i}", "response": "r"} for i in range(5)])
    src = SupervisedDataSource(train, eval_manifest_path=evalp, seed=0)
    batches = list(src.iter_eval_batches(2))
    assert [len(b) for b in batches] == [2, 2, 1]  # final partial batch yielded (trainer pads it)
    assert batches[0][0]["prompt"] == "e0"  # manifest order, no shuffle
    assert [len(b) for b in list(src.iter_eval_batches(2, eval_num_samples=3))] == [2, 1]
    assert list(src.iter_eval_batches(2, eval_num_samples=0)) == []


def test_cursor_rejects_shrunk_dataset(tmp_path):
    path = _write_manifest(tmp_path / "m.jsonl", [{"prompt": f"p{i}", "response": "r"} for i in range(4)])
    src = SupervisedDataSource(path, seed=0)
    with pytest.raises(ValueError, match="exceeds"):
        src.load_state_dict({"epoch": 0, "position": 9, "seed": 0})
