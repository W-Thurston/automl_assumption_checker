# app/report.py

import json
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.core.dispatcher import run_all_checks
from app.data.simulated_data import generate_linear_data


def generate_report(
    X,
    y,
    return_plot: bool = False,
    output_format: str = "console",
    verbose: bool = False,
):
    results = run_all_checks(X, y, return_plot=return_plot)

    if output_format == "console":
        print_console_report(results, verbose=verbose)
    elif output_format == "json":
        export_to_json(results)
    elif output_format == "markdown":
        export_to_markdown(results)
    else:
        raise ValueError("Unsupported output format")


def print_console_report(results, verbose: bool = False):
    console = Console()
    console.rule("[bold yellow]Assumption Check Report")
    for name, result in results.items():
        icon = "✅" if result.passed else "⚠️"
        panel_title = f"{icon} {name.title()}"
        recommendation = result.recommendation or "—"

        summary_text = result.summary.strip()

        # === Format Summary Line ===
        summary_text = result.summary.strip()
        if "→ Pass" in summary_text:
            summary_text = summary_text.replace("→ Pass", "→ [green]Pass[/green]")
        elif "→ Fail" in summary_text:
            summary_text = summary_text.replace("→ Fail", "→ [red]Fail[/red]")

        # === Format Details (Verbose Mode) ===
        metric_comparisons = {
            "r_squared": "≥",
            "breusch_pagan_pval": "≥",
            "shapiro_pval": "≥",
            "dagostino_pval": "≥",
            "anderson_stat": "≤",
            "vif": "≤",
        }
        # === Mapping between metrics and their threshold keys ===
        metric_threshold_pairs = {
            "r_squared": "r2_threshold",
            "breusch_pagan_pval": "homoscedasticity_pval_threshold",
            "shapiro_pval": "normality_pval_threshold",
            "dagostino_pval": "normality_pval_threshold",
            # Add others as needed
        }

        # (name, operator, values)
        # (name, operator, values)
        formatted_details = []

        for key, val in result.details.items():
            if key in metric_threshold_pairs:
                threshold_key = metric_threshold_pairs[key]
                threshold_val = result.details.get(threshold_key)
                if threshold_val is not None:
                    comp = metric_comparisons.get(key, "≥")
                    name_fmt = key.replace("_", " ").capitalize()
                    value_fmt = f"{val:.4f}"
                    threshold_fmt = f"{threshold_val:.4f}"
                    formatted_details.append((name_fmt, comp, value_fmt, threshold_fmt))
                    continue  # skip to next, don't double-append
            elif "threshold" not in key:
                name_fmt = key.replace("_", " ").capitalize()
                if isinstance(val, list):
                    value_fmt = "\n- " + "\n- ".join(val)
                elif isinstance(val, float):
                    value_fmt = f"{val:.4f}"
                else:
                    value_fmt = str(val)
                formatted_details.append((name_fmt, ":", value_fmt, None))

        # Find max label length
        label_width = max(len(name) for name, *_ in formatted_details)

        # Format all lines
        details_lines = []
        for name, op, val, threshold in formatted_details:
            padded_name = name.ljust(label_width)
            if threshold:
                line = (
                    f"{padded_name} {op} [cyan]Threshold[/cyan] "
                    f"| {val} {op} [cyan]{threshold}[/cyan]"
                )
            else:
                line = f"{padded_name} {op} {val}"
            details_lines.append(line)

        # === Severity Explanation ===
        severity_line = f"{result.severity} → " + {
            "low": "No meaningful concern",
            "moderate": "May impact model fit",
            "high": "Strong violation of assumption",
        }.get(result.severity, "Unknown")

        # === Build Rich Table ===
        table = Table.grid(padding=(0, 1))
        table.add_row("[bold]Summary:[/bold]", summary_text)
        if verbose and details_lines:
            table.add_row("[bold]Details:[/bold]", "\n".join(details_lines))
        table.add_row("[bold]Severity:[/bold]", severity_line)
        table.add_row("[bold]Recommendation:[/bold]", recommendation)

        # === Print Panel ===
        console.print(
            Panel(
                table,
                title=panel_title,
                border_style="green" if result.passed else "red",
            )
        )


def export_to_json(results, filename: str = None):
    payload = {k: r.__dict__ for k, r in results.items()}
    filename = (
        filename or f"assumption_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Report saved to {filename}")


def export_to_markdown(results, filename: str = None):
    lines = ["# Assumption Check Report\n"]
    for name, r in results.items():
        icon = "✅" if r.passed else "⚠️"
        lines.append(f"## {icon} {name.title()}")
        lines.append(f"**Summary:** {r.summary}")
        lines.append(f"**Severity:** {r.severity or 'unknown'}")
        if r.recommendation:
            lines.append(f"**Recommendation:** {r.recommendation}")
        lines.append("")  # spacing

    filename = (
        filename or f"assumption_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Markdown report saved to {filename}")


if __name__ == "__main__":
    df = generate_linear_data(seed=42)
    generate_report(
        df["x"], df["y"], return_plot=True, output_format="console", verbose=True
    )
