from __future__ import annotations

import json
from pathlib import Path


def write_json_report(records: list[dict], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, sort_keys=True))


def render_markdown_report(records: list[dict]) -> str:
    if not records:
        return "# Absence Modeling Report\n\nNo experiment results were generated.\n"

    lines = ["# Absence Modeling Report", ""]
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["experiment"], []).append(record)

    for experiment, experiment_records in grouped.items():
        lines.append(f"## {experiment}")
        lines.append("")
        for record in experiment_records:
            lines.append(f"### {record['task']} / {record['split']}")
            lines.append("")
            candidate = record.get("candidate", "n/a")
            lines.append(f"- model: `{candidate}`")
            lines.append(f"- rows: `{int(record.get('row_count', 0))}`")
            lines.append(f"- positive rate: `{record.get('positive_rate', float('nan')):.4f}`")
            if "pr_auc" in record:
                lines.append(f"- PR-AUC: `{record['pr_auc']:.4f}`")
            if "brier" in record:
                lines.append(f"- Brier: `{record['brier']:.4f}`")
            precision_keys = sorted(key for key in record if key.startswith("precision_at_"))
            for key in precision_keys:
                lines.append(f"- {key}: `{record[key]:.4f}`")
            lesson_keys = sorted(key for key in record if key.startswith("lesson_hit_rate_at_"))
            for key in lesson_keys:
                lines.append(f"- {key}: `{record[key]:.4f}`")
            if "rank_correlation" in record:
                lines.append(f"- rank_correlation: `{record['rank_correlation']:.4f}`")
            lines.append("")
    return "\n".join(lines)


def write_markdown_report(records: list[dict], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_report(records))
