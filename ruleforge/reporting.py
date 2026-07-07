"""
ruleforge/reporting.py
----------------------
Report generation — JSON, CSV, TSV, HTML, Markdown, and PDF output.

Generates structured reports from:
- Analysis results
- Generation runs
- Fitness statistics
- Runtime evaluation data
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """Container for a full RuleForge run report."""

    title: str = "RuleForge Report"
    timestamp: float = field(default_factory=time.time)
    config: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    top_rules: list[dict[str, Any]] = field(default_factory=list)  # {rule, score, origin}
    word_sampling: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "timestamp": self.timestamp,
            "config": self.config,
            "analysis": self.analysis,
            "generation": self.generation,
            "top_rules": self.top_rules,
            "word_sampling": self.word_sampling,
            "errors": self.errors,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class Reporter:
    """Generate reports in multiple formats.

    Args:
        report: The :class:`Report` to render.
    """

    def __init__(self, report: Report) -> None:
        self._report = report

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._report.to_dict(), indent=indent)

    def write_json(self, path: Path, indent: int = 2) -> None:
        path.write_text(self.to_json(indent=indent), encoding="utf-8")
        logger.info("Report written: %s", path)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def to_csv(self) -> str:
        """Return top-rules as CSV."""
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["rule", "score", "origin"],
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in self._report.top_rules:
            writer.writerow(row)
        return buf.getvalue()

    def write_csv(self, path: Path) -> None:
        path.write_text(self.to_csv(), encoding="utf-8")
        logger.info("CSV report written: %s", path)

    # ------------------------------------------------------------------
    # TSV
    # ------------------------------------------------------------------

    def to_tsv(self) -> str:
        """Return top-rules as TSV."""
        lines = ["rule\tscore\torigin\tround"]
        for row in self._report.top_rules:
            lines.append(
                f"{row.get('rule', '')}\t"
                f"{row.get('score', 0.0):.6f}\t"
                f"{row.get('origin', '')}\t"
                f"{row.get('round', '')}"
            )
        return "\n".join(lines)

    def write_tsv(self, path: Path) -> None:
        path.write_text(self.to_tsv(), encoding="utf-8")
        logger.info("TSV report written: %s", path)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        r = self._report
        gen = r.generation
        analysis = r.analysis
        lines = [
            f"# {r.title}",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r.timestamp))}",
            "",
            "## Analysis",
            "",
            f"- Total lines: {analysis.get('total_lines', 'N/A')}",
            f"- Valid rules: {analysis.get('valid_lines', 'N/A')}",
            f"- Unique rules: {analysis.get('unique_count', 'N/A')}",
            f"- Invalid rules: {analysis.get('invalid_lines', 'N/A')}",
            f"- Entropy: {analysis.get('entropy', 0.0):.4f}",
            "",
            "## Generation",
            "",
            f"- Rules kept: {gen.get('kept_rules', 'N/A')}",
            f"- Rounds: {gen.get('rounds', 'N/A')}",
            f"- Worker errors: {gen.get('worker_errors', 0)}",
            "",
            "## Top 20 Rules",
            "",
            "| # | Rule | Score | Origin |",
            "|---|------|-------|--------|",
        ]
        for i, row in enumerate(r.top_rules[:20], start=1):
            lines.append(
                f"| {i} | `{row.get('rule', '')}` "
                f"| {row.get('score', 0.0):.4f} "
                f"| {row.get('origin', '')} |"
            )

        if r.errors:
            lines += ["", "## Errors", ""]
            for err in r.errors[:20]:
                lines.append(f"- {err}")

        return "\n".join(lines)

    def write_markdown(self, path: Path) -> None:
        path.write_text(self.to_markdown(), encoding="utf-8")
        logger.info("Markdown report written: %s", path)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def to_html(self) -> str:
        r = self._report
        gen = r.generation
        analysis = r.analysis
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp))

        rows_html = ""
        for i, row in enumerate(r.top_rules[:100], start=1):
            rows_html += (
                f"<tr><td>{i}</td>"
                f"<td><code>{_html_escape(row.get('rule', ''))}</code></td>"
                f"<td>{row.get('score', 0.0):.4f}</td>"
                f"<td>{_html_escape(row.get('origin', ''))}</td></tr>\n"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_html_escape(r.title)}</title>
<style>
  body {{ font-family: monospace; max-width: 1200px; margin: 0 auto; padding: 1rem; }}
  h1, h2 {{ border-bottom: 1px solid #ccc; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: left; }}
  th {{ background: #f0f0f0; }}
  tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>{_html_escape(r.title)}</h1>
<p>Generated: {ts}</p>
<h2>Analysis</h2>
<ul>
  <li>Total lines: {analysis.get('total_lines', 'N/A')}</li>
  <li>Valid rules: {analysis.get('valid_lines', 'N/A')}</li>
  <li>Unique rules: {analysis.get('unique_count', 'N/A')}</li>
  <li>Invalid rules: {analysis.get('invalid_lines', 'N/A')}</li>
  <li>Entropy: {analysis.get('entropy', 0.0):.4f}</li>
</ul>
<h2>Generation</h2>
<ul>
  <li>Rules kept: {gen.get('kept_rules', 'N/A')}</li>
  <li>Rounds: {gen.get('rounds', 'N/A')}</li>
  <li>Worker errors: {gen.get('worker_errors', 0)}</li>
</ul>
<h2>Top Rules</h2>
<table>
<tr><th>#</th><th>Rule</th><th>Score</th><th>Origin</th></tr>
{rows_html}
</table>
</body>
</html>
"""

    def write_html(self, path: Path) -> None:
        path.write_text(self.to_html(), encoding="utf-8")
        logger.info("HTML report written: %s", path)

    # ------------------------------------------------------------------
    # PDF (optional — requires reportlab)
    # ------------------------------------------------------------------

    def write_pdf(self, path: Path) -> None:
        """Write a PDF report. Requires ``reportlab``."""
        try:
            from reportlab.lib.pagesizes import A4  # type: ignore[import]
            from reportlab.platypus import (  # type: ignore[import]
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
            )
            from reportlab.lib.styles import getSampleStyleSheet  # type: ignore[import]
            from reportlab.lib import colors  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PDF generation requires 'reportlab'. Install with: pip install reportlab"
            ) from exc

        r = self._report
        doc = SimpleDocTemplate(str(path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(r.title, styles["Title"]))
        story.append(Spacer(1, 12))

        gen = r.generation
        analysis = r.analysis
        story.append(Paragraph("Analysis", styles["Heading2"]))
        for k, v in analysis.items():
            story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Top Rules", styles["Heading2"]))
        table_data = [["#", "Rule", "Score", "Origin"]]
        for i, row in enumerate(r.top_rules[:50], start=1):
            table_data.append([
                str(i),
                str(row.get("rule", "")),
                f"{row.get('score', 0.0):.4f}",
                str(row.get("origin", "")),
            ])
        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(tbl)
        doc.build(story)
        logger.info("PDF report written: %s", path)

    # ------------------------------------------------------------------
    # Multi-format write
    # ------------------------------------------------------------------

    def write_all(self, directory: Path, stem: str = "report") -> None:
        """Write JSON, CSV, TSV, Markdown, and HTML reports."""
        directory.mkdir(parents=True, exist_ok=True)
        self.write_json(directory / f"{stem}.json")
        self.write_csv(directory / f"{stem}.csv")
        self.write_tsv(directory / f"{stem}.tsv")
        self.write_markdown(directory / f"{stem}.md")
        self.write_html(directory / f"{stem}.html")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
