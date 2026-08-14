"""Nightly Report Generation for Agent Analytics.

Generates daily and weekly summary reports from usage logs.
Outputs JSON and Markdown reports for monitoring.

Schedule with cron/Task Scheduler:
    0 6 * * * python analytics.py --daily
    0 6 * * 1 python analytics.py --weekly

Usage:
    python analytics.py --daily           # Yesterday's report
    python analytics.py --weekly          # Last 7 days report
    python analytics.py --output reports/ # Custom output directory
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bin.admin_tools import (
    get_usage_summary,
    get_model_distribution,
    get_tool_usage,
    get_recent_errors,
    get_top_queries,
    get_hourly_usage,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Report Generation
# =============================================================================

def generate_daily_report(date: Optional[str] = None) -> dict:
    """Generate a daily usage report.
    
    Args:
        date: Date string (YYYY-MM-DD). Defaults to yesterday.
    
    Returns:
        Report dict with all metrics
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    report = {
        "report_type": "daily",
        "date": date,
        "generated_at": datetime.now().isoformat(),
        "summary": get_usage_summary(days=1),
        "models": get_model_distribution(days=1),
        "tools": get_tool_usage(days=1),
        "errors": get_recent_errors(limit=20),
        "top_queries": get_top_queries(limit=10, days=1),
        "hourly": get_hourly_usage(days=1),
    }
    
    return report


def generate_weekly_report() -> dict:
    """Generate a weekly usage report.
    
    Returns:
        Report dict with 7-day metrics
    """
    report = {
        "report_type": "weekly",
        "period": f"{(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
        "generated_at": datetime.now().isoformat(),
        "summary": get_usage_summary(days=7),
        "models": get_model_distribution(days=7),
        "tools": get_tool_usage(days=7),
        "errors": get_recent_errors(limit=50),
        "top_queries": get_top_queries(limit=20, days=7),
    }
    
    return report


# =============================================================================
# Report Formatting
# =============================================================================

def report_to_markdown(report: dict) -> str:
    """Convert a report dict to readable Markdown."""
    lines = []
    
    # Header
    if report["report_type"] == "daily":
        lines.append(f"# Daily Report — {report['date']}")
    else:
        lines.append(f"# Weekly Report — {report['period']}")
    
    lines.append(f"\n_Generated: {report['generated_at'][:19]}_\n")
    
    # Summary
    s = report["summary"]
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Requests | {s.get('total_requests', 0):,} |")
    lines.append(f"| Unique Sessions | {s.get('unique_sessions', 0)} |")
    lines.append(f"| Total Tokens | {s.get('total_tokens', 0):,} |")
    lines.append(f"| Avg Latency | {s.get('avg_latency_ms', 0):.0f} ms |")
    
    # Models
    if report["models"]:
        lines.append("\n## Model Usage\n")
        lines.append("| Model | Requests | Tokens | Avg Latency |")
        lines.append("|-------|----------|--------|-------------|")
        for m in report["models"]:
            lines.append(
                f"| {m['model']} | {m['request_count']} | "
                f"{m.get('total_tokens', 0):,} | {m.get('avg_latency_ms', 0)} ms |"
            )
    
    # Tools
    if report["tools"]:
        lines.append("\n## Tool Usage\n")
        lines.append("| Tool | Calls | Avg Time | Success % |")
        lines.append("|------|-------|----------|-----------|")
        for t in report["tools"]:
            lines.append(
                f"| {t['tool_name']} | {t['call_count']} | "
                f"{t.get('avg_execution_ms', 0)} ms | {t.get('success_rate', 0)}% |"
            )
    
    # Top Queries
    if report["top_queries"]:
        lines.append("\n## Top Queries\n")
        for i, q in enumerate(report["top_queries"][:10], 1):
            lines.append(f"{i}. ({q['count']}x) {q['user_message'][:80]}")
    
    # Errors
    if report["errors"]:
        lines.append(f"\n## Errors ({len(report['errors'])})\n")
        for e in report["errors"][:10]:
            lines.append(
                f"- `{e.get('timestamp', '')[:19]}` — "
                f"**{e.get('error_type', '')}**: {e.get('error_message', '')[:60]}"
            )
    
    return "\n".join(lines)


# =============================================================================
# File Output
# =============================================================================

def save_report(report: dict, output_dir: str = "reports") -> tuple[Path, Path]:
    """Save report as JSON and Markdown.
    
    Args:
        report: Report dict
        output_dir: Output directory
    
    Returns:
        Tuple of (json_path, md_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Filename from type + date
    if report["report_type"] == "daily":
        basename = f"report_daily_{report['date']}"
    else:
        basename = f"report_weekly_{datetime.now().strftime('%Y-%m-%d')}"
    
    # JSON
    json_path = out / f"{basename}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Markdown
    md_path = out / f"{basename}.md"
    with open(md_path, "w") as f:
        f.write(report_to_markdown(report))
    
    logger.info(f"Report saved: {json_path}, {md_path}")
    return json_path, md_path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Generate usage analytics reports")
    parser.add_argument("--daily", action="store_true", help="Generate daily report (yesterday)")
    parser.add_argument("--weekly", action="store_true", help="Generate weekly report (last 7 days)")
    parser.add_argument("--output", default="reports", help="Output directory (default: reports/)")
    parser.add_argument("--print", action="store_true", dest="print_report", help="Print to stdout")
    args = parser.parse_args()
    
    if not args.daily and not args.weekly:
        args.daily = True  # Default to daily
    
    reports = []
    
    if args.daily:
        report = generate_daily_report()
        reports.append(report)
    
    if args.weekly:
        report = generate_weekly_report()
        reports.append(report)
    
    for report in reports:
        json_path, md_path = save_report(report, output_dir=args.output)
        print(f"  Saved: {json_path}")
        print(f"  Saved: {md_path}")
        
        if args.print_report:
            print("\n" + report_to_markdown(report))
