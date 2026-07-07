"""
ruleforge/plugins.py
--------------------
Plugin system for RuleForge.

Plugin types:
  Analyzer     — custom rule/password analysis
  Generator    — custom rule generation strategy
  Scorer       — custom scoring function
  Exporter     — custom output format
  Importer     — custom input format
  Runtime      — custom hashcat-like evaluation backend
  PasswordAnalyzer — custom password feature extractor
  Grammar      — custom grammar learner
  Mask         — custom mask generator
  Visualization — custom chart generator
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin type constants
# ---------------------------------------------------------------------------

class PluginType:
    ANALYZER = "analyzer"
    GENERATOR = "generator"
    SCORER = "scorer"
    EXPORTER = "exporter"
    IMPORTER = "importer"
    RUNTIME = "runtime"
    PASSWORD_ANALYZER = "password_analyzer"
    GRAMMAR = "grammar"
    MASK = "mask"
    VISUALIZATION = "visualization"


ALL_PLUGIN_TYPES = frozenset({
    PluginType.ANALYZER,
    PluginType.GENERATOR,
    PluginType.SCORER,
    PluginType.EXPORTER,
    PluginType.IMPORTER,
    PluginType.RUNTIME,
    PluginType.PASSWORD_ANALYZER,
    PluginType.GRAMMAR,
    PluginType.MASK,
    PluginType.VISUALIZATION,
})


# ---------------------------------------------------------------------------
# Plugin protocols (structural typing)
# ---------------------------------------------------------------------------

@runtime_checkable
class AnalyzerPlugin(Protocol):
    """Structural protocol for Analyzer plugins."""

    @property
    def name(self) -> str: ...

    def analyze(self, rules: list[str]) -> dict[str, Any]: ...


@runtime_checkable
class GeneratorPlugin(Protocol):
    """Structural protocol for Generator plugins."""

    @property
    def name(self) -> str: ...

    def generate(self, n: int) -> list[str]: ...


@runtime_checkable
class ScorerPlugin(Protocol):
    """Structural protocol for Scorer plugins."""

    @property
    def name(self) -> str: ...

    def score(self, rule: str) -> float: ...


@runtime_checkable
class ExporterPlugin(Protocol):
    """Structural protocol for Exporter plugins."""

    @property
    def name(self) -> str: ...

    @property
    def extension(self) -> str: ...

    def export(self, rules: list[dict[str, Any]], path: Path) -> None: ...


# ---------------------------------------------------------------------------
# Plugin registration entry
# ---------------------------------------------------------------------------


@dataclass
class PluginEntry:
    """Metadata + instance for a registered plugin."""

    name: str
    plugin_type: str
    instance: Any
    path: Path | None = None
    version: str = "0.0.1"
    description: str = ""


# ---------------------------------------------------------------------------
# Plugin Registry
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Central registry for all RuleForge plugins.

    Usage::

        registry = PluginRegistry()
        registry.register(PluginType.SCORER, my_scorer, name="custom_scorer")
        scorer = registry.get(PluginType.SCORER, "custom_scorer")
    """

    def __init__(self) -> None:
        self._plugins: dict[str, dict[str, PluginEntry]] = {
            pt: {} for pt in ALL_PLUGIN_TYPES
        }

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        plugin_type: str,
        instance: Any,
        *,
        name: str | None = None,
        path: Path | None = None,
        version: str = "0.0.1",
        description: str = "",
    ) -> None:
        """Register *instance* as a plugin of *plugin_type*.

        Args:
            plugin_type: One of the :class:`PluginType` constants.
            instance:    The plugin object.
            name:        Plugin name (falls back to ``instance.name`` or class name).
            path:        Source file path (for loaded plugins).
            version:     Plugin version string.
            description: Human-readable description.
        """
        if plugin_type not in ALL_PLUGIN_TYPES:
            raise ValueError(
                f"Unknown plugin type {plugin_type!r}. "
                f"Must be one of: {sorted(ALL_PLUGIN_TYPES)}"
            )
        pname = name or getattr(instance, "name", None) or type(instance).__name__
        entry = PluginEntry(
            name=pname,
            plugin_type=plugin_type,
            instance=instance,
            path=path,
            version=version,
            description=description,
        )
        self._plugins[plugin_type][pname] = entry
        logger.info("Plugin registered: type=%s name=%s", plugin_type, pname)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, plugin_type: str, name: str) -> Any | None:
        """Return the plugin instance for *plugin_type*/*name*, or ``None``."""
        entry = self._plugins.get(plugin_type, {}).get(name)
        return entry.instance if entry else None

    def list_plugins(self, plugin_type: str | None = None) -> list[PluginEntry]:
        """Return all registered plugins, optionally filtered by type."""
        if plugin_type:
            return list(self._plugins.get(plugin_type, {}).values())
        return [
            entry
            for bucket in self._plugins.values()
            for entry in bucket.values()
        ]

    def get_all(self, plugin_type: str) -> list[Any]:
        """Return all plugin instances of *plugin_type*."""
        return [e.instance for e in self._plugins.get(plugin_type, {}).values()]

    # ------------------------------------------------------------------
    # Dynamic loading
    # ------------------------------------------------------------------

    def load_from_file(self, path: Path) -> list[PluginEntry]:
        """Load plugins from a Python file.

        The file must define a ``register(registry)`` function that calls
        ``registry.register(...)`` for each plugin it provides.

        Args:
            path: Path to the plugin Python file.

        Returns:
            List of newly registered :class:`PluginEntry` objects.
        """
        before = {pt: set(self._plugins[pt].keys()) for pt in ALL_PLUGIN_TYPES}

        spec = importlib.util.spec_from_file_location("_ruleforge_plugin", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin from {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        if not hasattr(module, "register"):
            raise AttributeError(
                f"Plugin file {path} must define a 'register(registry)' function"
            )
        module.register(self)

        # Collect newly added entries
        new_entries: list[PluginEntry] = []
        for pt in ALL_PLUGIN_TYPES:
            for name in self._plugins[pt]:
                if name not in before[pt]:
                    entry = self._plugins[pt][name]
                    entry.path = path
                    new_entries.append(entry)

        logger.info("Loaded %d plugin(s) from %s", len(new_entries), path)
        return new_entries

    def load_from_directory(self, directory: Path) -> list[PluginEntry]:
        """Load all ``*.py`` plugin files from *directory*."""
        entries: list[PluginEntry] = []
        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                entries.extend(self.load_from_file(py_file))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load plugin %s: %s", py_file, exc)
        return entries

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            pt: [e.name for e in bucket.values()]
            for pt, bucket in self._plugins.items()
        }


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

_default_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Return the module-level default :class:`PluginRegistry`."""
    return _default_registry
