#!/usr/bin/env python3
"""Summarize token-cost JSON into human-readable tables.

Usage:
  cd benchmarks/lightmem
  uv run python summarize_token_cost.py \
    --input output/token_cost_lightmem.json \
    --format markdown
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class Row:
    model: str
    op_type: str
    calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_time_sec: float

    @property
    def avg_input_tokens(self) -> float:
        return self.input_tokens / self.calls if self.calls else 0.0

    @property
    def avg_output_tokens(self) -> float:
        return self.output_tokens / self.calls if self.calls else 0.0

    @property
    def avg_total_tokens(self) -> float:
        return self.total_tokens / self.calls if self.calls else 0.0

    @property
    def avg_time_sec(self) -> float:
        return self.total_time_sec / self.calls if self.calls else 0.0


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, bool):
            return default
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if isinstance(v, bool):
            return default
        if v is None:
            return default
        out = float(v)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _parse_cost_state(model: str, op_type: str, state: dict[str, Any]) -> Row:
    calls = _as_int(state.get("total_count"), 0)
    input_tokens = _as_int(state.get("input_tokens"), 0)
    output_tokens = _as_int(state.get("output_tokens"), 0)
    total_time_sec = _as_float(state.get("total_time"), 0.0)
    total_tokens = _as_int(state.get("total_tokens"), input_tokens + output_tokens)
    return Row(
        model=model,
        op_type=op_type,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        total_time_sec=total_time_sec,
    )


def parse_rows(data: dict[str, Any]) -> list[Row]:
    rows: list[Row] = []
    for model, model_state in data.items():
        if not isinstance(model_state, dict):
            continue

        if "histories" in model_state or "total_count" in model_state:
            rows.append(_parse_cost_state(model, "all", model_state))
            continue

        for op_type, op_state in model_state.items():
            if not isinstance(op_state, dict):
                continue
            rows.append(_parse_cost_state(model, str(op_type), op_state))

    return rows


def aggregate_by_model(rows: list[Row]) -> list[Row]:
    grouped: dict[str, Row] = {}
    for row in rows:
        if row.model not in grouped:
            grouped[row.model] = Row(
                model=row.model,
                op_type="ALL_OPS",
                calls=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                total_time_sec=0.0,
            )
        agg = grouped[row.model]
        agg.calls += row.calls
        agg.input_tokens += row.input_tokens
        agg.output_tokens += row.output_tokens
        agg.total_tokens += row.total_tokens
        agg.total_time_sec += row.total_time_sec
    return sorted(grouped.values(), key=lambda r: r.model)


def fmt_float(v: float) -> str:
    return f"{v:.4f}"


def to_markdown(rows: list[Row], title: str) -> str:
    header = (
        "| model | op_type | calls | input_tokens | output_tokens | total_tokens | "
        "avg_input | avg_output | avg_total | total_time_sec | avg_time_sec |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [f"### {title}", "", header]
    for r in rows:
        lines.append(
            f"| {r.model} | {r.op_type} | {r.calls} | {r.input_tokens} | {r.output_tokens} | "
            f"{r.total_tokens} | {fmt_float(r.avg_input_tokens)} | {fmt_float(r.avg_output_tokens)} | "
            f"{fmt_float(r.avg_total_tokens)} | {fmt_float(r.total_time_sec)} | {fmt_float(r.avg_time_sec)} |"
        )
    if len(rows) == 0:
        lines.append("| _no data_ | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
    return "\n".join(lines)


def to_tsv(rows: list[Row]) -> str:
    header = [
        "model",
        "op_type",
        "calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "avg_input",
        "avg_output",
        "avg_total",
        "total_time_sec",
        "avg_time_sec",
    ]
    lines = ["\t".join(header)]
    for r in rows:
        lines.append(
            "\t".join(
                [
                    r.model,
                    r.op_type,
                    str(r.calls),
                    str(r.input_tokens),
                    str(r.output_tokens),
                    str(r.total_tokens),
                    fmt_float(r.avg_input_tokens),
                    fmt_float(r.avg_output_tokens),
                    fmt_float(r.avg_total_tokens),
                    fmt_float(r.total_time_sec),
                    fmt_float(r.avg_time_sec),
                ]
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize token cost JSON to a table.")
    parser.add_argument(
        "--input",
        type=str,
        default="output/token_cost_lightmem.json",
        help="Path to token cost JSON.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "tsv"],
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output file path. If omitted, prints to stdout.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        print("ERROR: input JSON root must be an object.", file=sys.stderr)
        return 2

    detailed_rows = sorted(parse_rows(data), key=lambda r: (r.model, r.op_type))
    summary_rows = aggregate_by_model(detailed_rows)

    if args.format == "markdown":
        output = "\n\n".join(
            [
                to_markdown(detailed_rows, "Detailed By Model + OpType"),
                to_markdown(summary_rows, "Summary By Model"),
            ]
        )
    else:
        output = (
            "# Detailed By Model + OpType\n"
            + to_tsv(detailed_rows)
            + "\n\n# Summary By Model\n"
            + to_tsv(summary_rows)
        )

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(output + ("\n" if not output.endswith("\n") else ""))
        print(f"Saved summary to: {args.save}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
