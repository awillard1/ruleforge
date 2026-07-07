"""
ruleforge/checkpoint.py
-----------------------
Job checkpointing and resume support.

Saves and restores the full state of an interrupted generation job:
- Generator population / state
- Markov / N-gram model snapshots
- Random seeds
- Statistics
- Database state reference
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint data model
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """Complete state snapshot of a generation job.

    Attributes:
        job_id:        Unique job identifier.
        checkpoint_id: Auto-incremented checkpoint counter.
        timestamp:     Unix timestamp when checkpoint was created.
        config:        Job configuration dict.
        stats:         Aggregated statistics at checkpoint time.
        population:    List of ``{rule, score, origin}`` dicts.
        models:        Dict of serialized model data blobs.
        random_seeds:  Dict of component → seed value.
        extra:         Any additional state data.
    """

    job_id: str
    checkpoint_id: int = 0
    timestamp: float = field(default_factory=time.time)
    config: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    population: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)
    random_seeds: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "stats": self.stats,
            "population": self.population,
            "models": self.models,
            "random_seeds": self.random_seeds,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Checkpoint":
        return cls(
            job_id=d["job_id"],
            checkpoint_id=int(d.get("checkpoint_id", 0)),
            timestamp=float(d.get("timestamp", 0.0)),
            config=d.get("config", {}),
            stats=d.get("stats", {}),
            population=d.get("population", []),
            models=d.get("models", {}),
            random_seeds=d.get("random_seeds", {}),
            extra=d.get("extra", {}),
        )


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manage checkpoint creation, storage and loading.

    Checkpoints are stored as JSON files in *directory*:
      ``{directory}/{job_id}/checkpoint_{id:06d}.json``

    A ``latest.json`` symlink / copy always points to the most recent
    checkpoint for fast resume.

    Args:
        directory:    Base directory for checkpoint storage.
        keep_last:    Number of most-recent checkpoints to retain (0 = keep all).
    """

    def __init__(self, directory: Path, keep_last: int = 5) -> None:
        self._base = directory
        self._keep_last = keep_last

    def _job_dir(self, job_id: str) -> Path:
        return self._base / job_id

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, checkpoint: Checkpoint) -> Path:
        """Write *checkpoint* to disk and update the ``latest`` pointer.

        Returns the path of the saved file.
        """
        job_dir = self._job_dir(checkpoint.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        cp_file = job_dir / f"checkpoint_{checkpoint.checkpoint_id:06d}.json"
        cp_file.write_text(
            json.dumps(checkpoint.to_dict(), indent=2),
            encoding="utf-8",
        )

        # Update latest pointer (overwrite)
        latest = job_dir / "latest.json"
        latest.write_text(
            json.dumps(checkpoint.to_dict(), indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Checkpoint saved: job=%s id=%d path=%s",
            checkpoint.job_id,
            checkpoint.checkpoint_id,
            cp_file,
        )

        self._prune(checkpoint.job_id)
        return cp_file

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_latest(self, job_id: str) -> Checkpoint | None:
        """Load the most recent checkpoint for *job_id*.

        Returns ``None`` if no checkpoint exists.
        """
        latest = self._job_dir(job_id) / "latest.json"
        if not latest.exists():
            logger.info("No checkpoint found for job %s", job_id)
            return None
        data = json.loads(latest.read_text(encoding="utf-8"))
        cp = Checkpoint.from_dict(data)
        logger.info(
            "Checkpoint loaded: job=%s id=%d",
            cp.job_id,
            cp.checkpoint_id,
        )
        return cp

    def load(self, path: Path) -> Checkpoint:
        """Load a specific checkpoint file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return Checkpoint.from_dict(data)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_checkpoints(self, job_id: str) -> list[Path]:
        """Return sorted list of checkpoint files for *job_id*."""
        job_dir = self._job_dir(job_id)
        if not job_dir.exists():
            return []
        files = sorted(job_dir.glob("checkpoint_*.json"))
        return files

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------

    def _prune(self, job_id: str) -> None:
        """Remove old checkpoints if *keep_last* is set."""
        if self._keep_last <= 0:
            return
        files = self.list_checkpoints(job_id)
        if len(files) > self._keep_last:
            for old in files[: -self._keep_last]:
                old.unlink(missing_ok=True)
                logger.debug("Pruned checkpoint: %s", old)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_all(self, job_id: str) -> None:
        """Remove all checkpoints for *job_id*."""
        import shutil
        job_dir = self._job_dir(job_id)
        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info("All checkpoints deleted for job %s", job_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, job_id: str) -> dict[str, Any]:
        files = self.list_checkpoints(job_id)
        return {
            "job_id": job_id,
            "checkpoint_count": len(files),
            "latest": str(files[-1]) if files else None,
        }
