"""
ruleforge/visualization.py
--------------------------
Visualization — generate charts and graphs from analysis data.

Charts generated:
- Operation frequency histogram
- Rule length distribution
- Transition graph (Markov)
- Fitness history graph (evolution)
- Coverage graph
- Markov probability graph
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _require_matplotlib() -> Any:
    """Import matplotlib and return the pyplot module, or raise ImportError."""
    try:
        import matplotlib  # type: ignore[import]
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt  # type: ignore[import]
        return plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires 'matplotlib'. Install with: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


class Visualizer:
    """Generate and save visualizations from RuleForge analysis data.

    Args:
        output_dir: Directory to write chart files to.
        dpi:        Image resolution (dots per inch).
        fmt:        Image format (``"png"`` or ``"svg"``).
    """

    def __init__(
        self,
        output_dir: Path,
        dpi: int = 120,
        fmt: str = "png",
    ) -> None:
        self._dir = output_dir
        self._dpi = dpi
        self._fmt = fmt
        self._dir.mkdir(parents=True, exist_ok=True)

    def _save(self, plt: Any, name: str) -> Path:
        path = self._dir / f"{name}.{self._fmt}"
        plt.savefig(str(path), dpi=self._dpi, bbox_inches="tight")
        plt.close()
        logger.info("Chart saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Operation frequency histogram
    # ------------------------------------------------------------------

    def cmd_histogram(self, cmd_freq: dict[str, int], title: str = "Operation Frequency") -> Path:
        """Bar chart of operation frequencies."""
        plt = _require_matplotlib()
        from .parser import Parser as _P  # local import to avoid circular
        _p = _P()

        labels = sorted(cmd_freq.keys(), key=lambda c: cmd_freq[c], reverse=True)
        values = [cmd_freq[c] for c in labels]
        display = [f"{c}\n{_p.op_name(c)[:6]}" for c in labels]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.4), 5))
        ax.bar(display, values, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Operation")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelsize=7)
        plt.tight_layout()
        return self._save(plt, "cmd_histogram")

    # ------------------------------------------------------------------
    # Rule length distribution
    # ------------------------------------------------------------------

    def length_distribution(self, len_dist: dict[int, int]) -> Path:
        """Bar chart of rule length distribution."""
        plt = _require_matplotlib()
        lengths = sorted(int(k) for k in len_dist.keys())
        counts = [len_dist[str(k)] if str(k) in len_dist else len_dist.get(k, 0) for k in lengths]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([str(l) for l in lengths], counts, color="coral")
        ax.set_title("Rule Length Distribution")
        ax.set_xlabel("Number of Operations")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return self._save(plt, "length_distribution")

    # ------------------------------------------------------------------
    # Transition graph
    # ------------------------------------------------------------------

    def transition_graph(self, trans: dict[str, dict[str, float]], top_n: int = 15) -> Path:
        """Directed graph of operation transitions (requires networkx)."""
        plt = _require_matplotlib()
        try:
            import networkx as nx  # type: ignore[import]
        except ImportError:
            logger.warning("networkx not installed; skipping transition graph")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "networkx not installed", ha="center", va="center")
            return self._save(plt, "transition_graph")

        G = nx.DiGraph()
        edges: list[tuple[str, str, float]] = []
        for src, dsts in trans.items():
            for dst, prob in dsts.items():
                edges.append((src, dst, prob))
        edges.sort(key=lambda e: e[2], reverse=True)
        for src, dst, prob in edges[:top_n * 2]:
            G.add_edge(src, dst, weight=prob)

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        weights = [G[u][v]["weight"] * 5 for u, v in G.edges()]
        nx.draw_networkx(
            G, pos, ax=ax,
            node_size=600,
            font_size=9,
            width=weights,
            edge_color="gray",
            arrows=True,
        )
        ax.set_title("Operation Transition Graph")
        ax.axis("off")
        plt.tight_layout()
        return self._save(plt, "transition_graph")

    # ------------------------------------------------------------------
    # Fitness history
    # ------------------------------------------------------------------

    def fitness_history(
        self,
        best_per_gen: list[float],
        mean_per_gen: list[float] | None = None,
    ) -> Path:
        """Line chart of fitness over generations."""
        plt = _require_matplotlib()
        fig, ax = plt.subplots(figsize=(10, 5))
        generations = list(range(len(best_per_gen)))
        ax.plot(generations, best_per_gen, label="Best", color="green")
        if mean_per_gen:
            ax.plot(generations, mean_per_gen, label="Mean", color="blue", linestyle="--")
        ax.set_title("Fitness History (Evolution)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend()
        plt.tight_layout()
        return self._save(plt, "fitness_history")

    # ------------------------------------------------------------------
    # Coverage graph
    # ------------------------------------------------------------------

    def coverage_graph(self, coverage_over_time: list[float]) -> Path:
        """Line chart of baseline coverage growth."""
        plt = _require_matplotlib()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(list(range(len(coverage_over_time))), coverage_over_time, color="purple")
        ax.set_title("Coverage Over Time")
        ax.set_xlabel("Rules Evaluated")
        ax.set_ylabel("Unique Outputs")
        plt.tight_layout()
        return self._save(plt, "coverage_graph")

    # ------------------------------------------------------------------
    # Markov probability distribution
    # ------------------------------------------------------------------

    def markov_prob_histogram(self, scores: list[float], title: str = "Markov Scores") -> Path:
        """Histogram of Markov probability scores."""
        plt = _require_matplotlib()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(scores, bins=40, color="teal", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        return self._save(plt, "markov_prob_histogram")

    # ------------------------------------------------------------------
    # Convenience: generate all standard charts from an analysis result
    # ------------------------------------------------------------------

    def generate_all(self, analysis_dict: dict[str, Any]) -> list[Path]:
        """Generate all standard charts from an :class:`~ruleforge.analyzer.AnalysisResult` dict."""
        paths: list[Path] = []

        if cmd_freq := analysis_dict.get("cmd_freq"):
            try:
                paths.append(self.cmd_histogram(cmd_freq))
            except Exception as exc:  # noqa: BLE001
                logger.warning("cmd_histogram failed: %s", exc)

        if len_dist := analysis_dict.get("len_dist"):
            try:
                # Normalize keys to int
                ld = {int(k): v for k, v in len_dist.items()}
                paths.append(self.length_distribution(ld))  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                logger.warning("length_distribution failed: %s", exc)

        if trans_probs := analysis_dict.get("transition_probs"):
            try:
                paths.append(self.transition_graph(trans_probs))
            except Exception as exc:  # noqa: BLE001
                logger.warning("transition_graph failed: %s", exc)

        return paths
