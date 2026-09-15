"""CLI entry point: show_model_spec.

Prints the input contract of a checkpoint, read from the fiddle config bundled
with weathernext. Needs the `model` dependency group but no weights, no GPU and
no network access.

Usage:
    uv run python scripts/show_model_spec.py --config configs/krovanh.yaml
    uv run python scripts/show_model_spec.py --model WeatherNextCyclones_Mini --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from wn2_typhoon.config import load_raw
from wn2_typhoon.model_spec import ModelSpec, load_spec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    parser.add_argument(
        "--model",
        help="model name; defaults to model.name in the experiment config",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of a report"
    )
    return parser.parse_args()


def report(spec: ModelSpec, model_name: str) -> str:
    """Render a human-readable summary of one checkpoint's input contract.

    Args:
        spec: The derived spec.
        model_name: Model name, for the heading.

    Returns:
        The report text.
    """
    levels = list(spec.pressure_levels)
    lines = [
        f"{model_name}  ({spec.config_name})",
        f"  input duration   : {spec.input_duration} (2 frames at t-6h and t)",
        f"  pressure levels  : {len(levels)} hPa {levels}",
        "",
        "  CDS download",
    ]
    for group, variables in (
        ("pressure levels", spec.pressure_level_vars),
        ("single level", spec.single_level_vars),
        ("single level, static", spec.static_vars),
    ):
        lines.append(f"    {group} ({len(variables)}):")
        for variable in variables:
            source = spec.era5_source(variable)
            suffix = "" if variable == source.variable else f"  <- {source.variable}"
            lines.append(f"      {variable}{suffix}")
    computed = ", ".join(spec.computed_vars)
    cyclone_fields = sum(v.startswith("cyclone_") for v in spec.target_vars)
    lines += [
        "",
        f"  computed locally ({len(spec.computed_vars)}): {computed}",
        f"  targets: {len(spec.target_vars)} variables, "
        + f"of which {cyclone_fields} cyclone fields",
    ]
    return "\n".join(lines)


def main() -> None:
    """Print the input contract of the configured checkpoint."""
    args = parse_args()
    model_name = args.model or load_raw(args.config)["model"]["name"]
    spec = load_spec(model_name)
    if args.json:
        print(json.dumps(dataclasses.asdict(spec), indent=2))
    else:
        print(report(spec, model_name))


if __name__ == "__main__":
    main()
