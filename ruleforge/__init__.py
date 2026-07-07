"""RuleForge — Advanced Hashcat rule learning and optimization framework."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("ruleforge")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
