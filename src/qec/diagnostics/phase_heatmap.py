"""
v6.0.0 — ASCII Phase Heatmap Output.

Prints a compact ASCII heatmap of the decoder phase diagram for
CLI inspection.  Maps dominant phase values to symbols for quick
visual assessment of phase structure.

Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any


def print_phase_heatmap(phase_diagram: dict[str, Any]) -> str:
    """Print an ASCII heatmap of the phase diagram.

    Maps dominant phase values to symbols:
        +1 → success basin
         0 → boundary / metastable
        -1 → failure basin

    Parameters
    ----------
    phase_diagram : dict
        Phase diagram result from ``build_decoder_phase_diagram()``.
        Must contain ``"grid_axes"`` and ``"cells"``.

    Returns
    -------
    str
        The formatted heatmap string (also printed to stdout).
    """
    axes = phase_diagram["grid_axes"]
    cells = phase_diagram["cells"]

    x_name = axes["x_name"]
    y_name = axes["y_name"]
    x_values = axes["x_values"]
    y_values = axes["y_values"]

    # Build cell lookup.
    cell_lookup: dict[tuple, dict] = {}
    for c in cells:
        cell_lookup[(c["x"], c["y"])] = c

    # Phase symbol map.
    symbol_map = {1: "+1", 0: " 0", -1: "-1"}

    lines: list[str] = []
    lines.append("")
    lines.append("Phase diagram summary")
    lines.append("")
    lines.append(f"{y_name} \\u2193 / {x_name} \\u2192")
    lines.append("")

    # Header row with x values.
    header = f"{'':>8s}"
    for x in x_values:
        header += f" {x:>6}"
    lines.append(header)

    # Data rows.
    for y in y_values:
        row = f"{y:>8}"
        for x in x_values:
            cell = cell_lookup.get((x, y))
            if cell is not None:
                symbol = symbol_map.get(cell["dominant_phase"], " ?")
                row += f" {symbol:>6s}"
            else:
                row += f" {'--':>6s}"
        lines.append(row)

    lines.append("")
    lines.append("Legend:")
    lines.append("  +1 = success basin")
    lines.append("   0 = boundary / metastable")
    lines.append("  -1 = failure basin")
    lines.append("")

    output = "\n".join(lines)
    print(output)
    return output
