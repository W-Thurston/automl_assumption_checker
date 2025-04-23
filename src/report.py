# src/report.py
import argparse
import json
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.dispatcher import run_all_checks
from src.data.simulated_data import list_simulations


def generate_report(
    X,
    y,
    model_type=None,
    return_plot: bool = False,
    output_format: str = "console",
    verbose: bool = False,
):
    """
    Generate an assumption diagnostic report using the registered checks.

    Args:
        X (pd.Series or pd.DataFrame): Predictor values (1D or multivariate)
        y (pd.Series): Response values.
        return_plot (bool, optional): Include base64-encoded plots in results.
        output_format (str): 'console', 'json', or 'markdown'.
        verbose (bool): If True, includes extra detail in console output.

    Raises:
        ValueError: If the output_format is not recognized.
    """
    results, model_wrapper = run_all_checks(
        X, y, model_type=model_type, return_plot=return_plot
    )

    if output_format == "console":
        print_console_report(results, model_wrapper=model_wrapper, verbose=verbose)
    elif output_format == "json":
        export_to_json(results)
    elif output_format == "markdown":
        export_to_markdown(results)
    else:
        raise ValueError("Unsupported output format")


def print_console_report(results, model_wrapper, verbose: bool = False):
    """
    Print a structured Rich panel for each assumption result.

    Args:
        results (dict): Assumption names mapped to AssumptionResult objects.
        verbose (bool): If True, includes details like thresholds and comparisons.
    """
    console = Console()
    console.rule("[bold yellow]Assumption Check Report")

    # Print mdoel metadata
    model_info = model_wrapper.summary().get("model_type", "Unknown")
    console.print(f"[bold cyan]Model Type:[/bold cyan] {model_info}")

    for name, result in results.items():

        # Determine pass/fail icon and panel title
        icon = "✅" if result.passed else "⚠️"
        panel_title = f"{icon} {name.title()}"
        recommendation = result.recommendation or "—"

        # Format Summary Line
        summary_text = result.summary.strip()

        # Highlight pass/fail in summary with color
        if "→ Pass" in summary_text:
            summary_text = summary_text.replace("→ Pass", "→ [green]Pass[/green]")
        elif "→ Fail" in summary_text:
            summary_text = summary_text.replace("→ Fail", "→ [red]Fail[/red]")

        # Lookup comparison operator and threshold for each metric
        metric_comparisons = {
            "r_squared": "≥",
            "breusch_pagan_pval": "≥",
            "shapiro_pval": "≥",
            "dagostino_pval": "≥",
            "anderson_stat": "≤",
            "vif": "≤",
            "durbin_watson": "in",
            # Add others as needed
        }
        # Mapping between metrics and their threshold keys
        metric_threshold_pairs = {
            "r_squared": "r2_threshold",
            "breusch_pagan_pval": "homoscedasticity_pval_threshold",
            "shapiro_pval": "normality_pval_threshold",
            "dagostino_pval": "normality_pval_threshold",
            "durbin_watson": "expected_range",
            # Add others as needed
        }

        # Format each detail line with aligned label, operator, and threshold comparison
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
            # Match feature VIF + threshold pairs dynamically
            elif key.lower().endswith("(vif)"):
                feature = key[:-6].strip().lower()  # remove " (vif)"
                threshold_key = f"{feature} threshold"
                threshold_val = result.details.get(threshold_key)
                if threshold_val is not None:
                    comp = "≤"
                    name_fmt = key
                    value_fmt = f"{val:.4f}"
                    threshold_fmt = f"{threshold_val:.4f}"
                    formatted_details.append((name_fmt, comp, value_fmt, threshold_fmt))
                    continue
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

        # Add severity explanation (contextual guidance)
        severity_line = f"{result.severity} → " + {
            "low": "No meaningful concern",
            "moderate": "May impact model fit",
            "high": "Strong violation of assumption",
        }.get(result.severity, "Unknown")

        # Build Rich Table
        table = Table.grid(padding=(0, 1))
        table.add_row("[bold]Summary:[/bold]", summary_text)
        if verbose and details_lines:
            table.add_row("[bold]Details:[/bold]", "\n".join(details_lines))
        table.add_row("[bold]Severity:[/bold]", severity_line)
        table.add_row("[bold]Recommendation:[/bold]", recommendation)

        # Print Panel
        console.print(
            Panel(
                table,
                title=panel_title,
                border_style="green" if result.passed else "red",
            )
        )


def export_to_json(results, filename: str = None):
    payload = {k: r.__dict__ for k, r in results.items()}

    # Default to timestamped filename if none provided
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

    # Default to timestamped filename if none provided
    filename = (
        filename or f"assumption_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Markdown report saved to {filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run statistical assumption checks for supervised models."
    )
    parser.add_argument(
        "--data",
        choices=list_simulations().keys(),
        default="linear",
        help="Which simulated dataset to run assumption checks on.",
    )
    parser.add_argument(
        "--model-type",
        choices=["linear"],
        default="linear",
        help="Which model to fit for diagnostics.",
    )
    parser.add_argument(
        "--format",
        choices=["console", "json", "markdown"],
        default="console",
        help="Output format.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include detailed threshold info in report.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Include base64-encoded plots."
    )

    args = parser.parse_args()

    diagnostic_context = {
        "model_type": args.model_type,
    }

    data_func = list_simulations()[args.data]
    df = data_func(seed=42)

    # Choose predictors dynamically
    X = df.drop(columns="y")
    y = df["y"]

    generate_report(
        X,
        y,
        model_type=args.model_type,
        return_plot=args.plot,
        output_format=args.format,
        verbose=args.verbose,
    )
