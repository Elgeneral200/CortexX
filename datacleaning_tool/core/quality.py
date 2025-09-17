"""
Data Quality Module

Provides:
- Rule schema and human-readable labels
- Rule execution engine (vectorized checks)
- Summary KPIs
- HTML report generation (dark theme), no external templating dependencies

Author: CortexX Team
Version: 1.0.0
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Tuple

import pandas as pd


def rule_label(rule: Dict[str, Any]) -> str:
    rtype = rule.get("type")
    p = rule.get("params", {})

    if rtype == "not_null":
        return f"Not Null: {p.get('column')}"
    if rtype == "unique":
        return f"Unique: {p.get('column')}"
    if rtype == "unique_multi":
        return f"Unique across: {', '.join(p.get('columns', []))}"
    if rtype == "min":
        return f"Min {p.get('min')} on {p.get('column')}"
    if rtype == "max":
        return f"Max {p.get('max')} on {p.get('column')}"
    if rtype == "between":
        return f"Between [{p.get('min')}, {p.get('max')}] on {p.get('column')}"
    if rtype == "allowed":
        return f"Allowed set on {p.get('column')}"
    if rtype == "regex":
        return f"Regex '{p.get('pattern')}' on {p.get('column')}"
    if rtype == "dtype":
        return f"Dtype is {p.get('dtype')} on {p.get('column')}"

    return f"{rtype}: {p}"


def _check_not_null(df: pd.DataFrame, column: str) -> Tuple[pd.Series, Dict[str, Any]]:
    mask = df[column].notna()
    return mask, {"missing": int((~mask).sum())}


def _check_unique(df: pd.DataFrame, column: str) -> Tuple[pd.Series, Dict[str, Any]]:
    dup_mask = df[column].duplicated(keep=False) & df[column].notna()
    return ~dup_mask, {"duplicates": int(dup_mask.sum())}


def _check_unique_multi(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.Series, Dict[str, Any]]:
    dup_mask = df.duplicated(subset=columns, keep=False)
    return ~dup_mask, {"duplicates": int(dup_mask.sum())}


def _check_min(df: pd.DataFrame, column: str, min_val: Any) -> Tuple[pd.Series, Dict[str, Any]]:
    mask = pd.to_numeric(df[column], errors="coerce") >= min_val
    return mask, {}


def _check_max(df: pd.DataFrame, column: str, max_val: Any) -> Tuple[pd.Series, Dict[str, Any]]:
    mask = pd.to_numeric(df[column], errors="coerce") <= max_val
    return mask, {}


def _check_between(df: pd.DataFrame, column: str, min_val: Any, max_val: Any) -> Tuple[pd.Series, Dict[str, Any]]:
    num = pd.to_numeric(df[column], errors="coerce")
    mask = (num >= min_val) & (num <= max_val)
    return mask, {}


def _check_allowed(df: pd.DataFrame, column: str, allowed: List[Any]) -> Tuple[pd.Series, Dict[str, Any]]:
    mask = df[column].isin(allowed)
    return mask, {}


def _check_regex(df: pd.DataFrame, column: str, pattern: str) -> Tuple[pd.Series, Dict[str, Any]]:
    try:
        regex = re.compile(pattern)
    except re.error:
        mask = df[column].isna()
        return mask, {"error": "invalid regex"}

    series = df[column].astype(str).fillna("")
    mask = series.apply(lambda x: bool(regex.search(x)))
    return mask, {}


def _check_dtype(df: pd.DataFrame, column: str, dtype: str) -> Tuple[pd.Series, Dict[str, Any]]:
    actual = str(df[column].dtype)
    mask = pd.Series([actual == dtype] * len(df), index=df.index)
    return mask, {"actual": actual}


def run_rules(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    fail_indices_all: set = set()
    issue_columns: set = set()

    for rule in rules:
        rtype = rule.get("type")
        params = rule.get("params", {})
        label = rule_label(rule)

        try:
            if rtype == "not_null":
                mask, stats = _check_not_null(df, params["column"])
                cols = [params["column"]]
            elif rtype == "unique":
                mask, stats = _check_unique(df, params["column"])
                cols = [params["column"]]
            elif rtype == "unique_multi":
                mask, stats = _check_unique_multi(df, params["columns"])
                cols = params["columns"]
            elif rtype == "min":
                mask, stats = _check_min(df, params["column"], params["min"])
                cols = [params["column"]]
            elif rtype == "max":
                mask, stats = _check_max(df, params["column"], params["max"])
                cols = [params["column"]]
            elif rtype == "between":
                mask, stats = _check_between(df, params["column"], params["min"], params["max"])
                cols = [params["column"]]
            elif rtype == "allowed":
                mask, stats = _check_allowed(df, params["column"], params.get("allowed", []))
                cols = [params["column"]]
            elif rtype == "regex":
                mask, stats = _check_regex(df, params["column"], params.get("pattern", ""))
                cols = [params["column"]]
            elif rtype == "dtype":
                mask, stats = _check_dtype(df, params["column"], params.get("dtype", ""))
                cols = [params["column"]]
            else:
                mask = pd.Series([False] * len(df), index=df.index)
                stats = {}
                cols = []
        except Exception as e:
            mask = pd.Series([False] * len(df), index=df.index)
            stats = {"error": str(e)}
            cols = []

        failed = (~mask).fillna(True)
        failed_indices = df.index[failed].tolist()
        fail_indices_all.update(failed_indices)

        for c in cols:
            if len(failed_indices) > 0:
                issue_columns.add(c)

        results.append(
            {
                "label": label,
                "type": rtype,
                "params": params,
                "passed": failed.sum() == 0,
                "failed_count": int(failed.sum()),
                "failed_indices": failed_indices[:100],
                "stats": stats,
            }
        )

    passed_rules = sum(1 for r in results if r["passed"])
    pass_rate = (passed_rules / len(results) * 100) if results else 100.0
    summary = {"pass_rate": pass_rate, "failed_rows": len(fail_indices_all), "issue_columns": sorted(issue_columns)}

    return results, summary


def results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Rule": r["label"],
                "Type": r["type"],
                "Failed Rows": r["failed_count"],
                "Passed": "✅" if r["passed"] else "❌",
            }
            for r in results
        ]
    )


def generate_html_report(
    df: pd.DataFrame,
    rules: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    meta: Dict[str, Any],
) -> str:
    def esc(x: Any) -> str:
        return html.escape(str(x))

    rules_rows = "".join(
        [
            f"<tr><td>{i}</td><td>{esc(rule_label(r))}</td><td><code>{esc(r.get('params', {}))}</code></td></tr>"
            for i, r in enumerate(rules, 1)
        ]
    )

    results_rows = "".join(
        [
            f"<tr><td>{esc(r['label'])}</td><td>{esc(r['type'])}</td><td>{r['failed_count']}</td><td>{'PASS' if r['passed'] else 'FAIL'}</td></tr>"
            for r in results
        ]
    )

    fail_index_set = set()
    [fail_index_set.update(r["failed_indices"]) for r in results]
    fail_index_list = list(fail_index_set)[:50]
    sample_html = df.loc[fail_index_list].to_html(classes="table", border=0) if fail_index_list else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8" />
<title>Data Quality Report</title>
<style>
  :root {{ --bg-primary:#0b1220; --bg-panel:#0f172a; --bg-surface:#111827; --text:#e5e7eb; --muted:#94a3b8; --accent:#3b82f6; --border:#334155; }}
  body {{ background:var(--bg-primary); color:var(--text); font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,Cantarell,'Fira Sans','Droid Sans','Helvetica Neue',sans-serif; margin:0; padding:20px; }}
  .card {{ background:var(--bg-surface); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px; box-shadow:0 6px 12px rgba(0,0,0,0.25); }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
  .kpi {{ background:var(--bg-panel); border:1px solid var(--border); border-radius:8px; padding:12px; text-align:center; }}
  table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
  th,td {{ border:1px solid var(--border); padding:8px; color:var(--text); }} th {{ background:#0f172a; }}
  .muted {{ color:var(--muted); }}
</style></head><body>
  <div class="card"><h1>Data Quality Report</h1><p class="muted">Generated at: {esc(meta.get("generated_at", "N/A"))}</p></div>
  <div class="card"><h2>Summary</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="muted">Pass Rate</div><div style="font-size:1.4em;">{summary['pass_rate']:.1f}%</div></div>
      <div class="kpi"><div class="muted">Failed Rows</div><div style="font-size:1.4em;">{summary['failed_rows']}</div></div>
      <div class="kpi"><div class="muted">Columns with Issues</div><div style="font-size:1.4em;">{len(summary['issue_columns'])}</div></div>
    </div>
  </div>
  <div class="card"><h2>Rules</h2><table><thead><tr><th>#</th><th>Rule</th><th>Params</th></tr></thead><tbody>{rules_rows}</tbody></table></div>
  <div class="card"><h2>Results</h2><table><thead><tr><th>Rule</th><th>Type</th><th>Failed Rows</th><th>Status</th></tr></thead><tbody>{results_rows}</tbody></table></div>
  <div class="card"><h2>Sample of Failing Rows (up to 50)</h2>{sample_html if sample_html else '<p class="muted">No failing rows to display.</p>'}</div>
</body></html>"""
