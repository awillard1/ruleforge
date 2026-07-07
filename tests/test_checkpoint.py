"""Tests for ruleforge/checkpoint.py"""

import json
import pytest
from pathlib import Path
from ruleforge.checkpoint import Checkpoint, CheckpointManager


@pytest.fixture
def cp_dir(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture
def manager(cp_dir):
    return CheckpointManager(cp_dir, keep_last=3)


def make_checkpoint(job_id="job1", checkpoint_id=0):
    return Checkpoint(
        job_id=job_id,
        checkpoint_id=checkpoint_id,
        config={"workers": 4, "seed": 42},
        stats={"valid_lines": 1000},
        population=[{"rule": "l$1", "fitness": 0.5, "generation": 0}],
    )


class TestCheckpoint:
    def test_to_from_dict(self):
        cp = make_checkpoint()
        d = cp.to_dict()
        cp2 = Checkpoint.from_dict(d)
        assert cp2.job_id == cp.job_id
        assert cp2.config == cp.config
        assert cp2.population == cp.population


class TestCheckpointManager:
    def test_save_creates_file(self, manager, cp_dir):
        cp = make_checkpoint()
        path = manager.save(cp)
        assert path.exists()

    def test_save_creates_latest(self, manager, cp_dir):
        cp = make_checkpoint()
        manager.save(cp)
        latest = cp_dir / "job1" / "latest.json"
        assert latest.exists()

    def test_load_latest(self, manager):
        cp = make_checkpoint(checkpoint_id=5)
        manager.save(cp)
        loaded = manager.load_latest("job1")
        assert loaded is not None
        assert loaded.checkpoint_id == 5

    def test_load_latest_missing(self, manager):
        assert manager.load_latest("nonexistent_job") is None

    def test_list_checkpoints(self, manager):
        manager.save(make_checkpoint(checkpoint_id=0))
        manager.save(make_checkpoint(checkpoint_id=1))
        files = manager.list_checkpoints("job1")
        assert len(files) == 2

    def test_prune(self, manager, cp_dir):
        for i in range(5):
            manager.save(make_checkpoint(checkpoint_id=i))
        files = manager.list_checkpoints("job1")
        assert len(files) <= 3  # keep_last=3

    def test_delete_all(self, manager):
        manager.save(make_checkpoint())
        manager.delete_all("job1")
        assert not (manager._base / "job1").exists()

    def test_stats(self, manager):
        manager.save(make_checkpoint(checkpoint_id=0))
        s = manager.stats("job1")
        assert s["checkpoint_count"] >= 1

    def test_load_specific_file(self, manager, tmp_path):
        cp = make_checkpoint(checkpoint_id=99)
        path = manager.save(cp)
        loaded = manager.load(path)
        assert loaded.checkpoint_id == 99
