#!/usr/bin/env python3
"""Generate a C header from a CSR JSON map.

Usage:
    python tools/gen_csr_header.py --json graphics_pipeline_csr_map.json --out graphics_pipeline_csr.h

The generator emits:
- compile-time constants for data/address width and granularity
- #define macros for each register byte address
- an enum with the same byte addresses
- a packed struct reflecting the register layout (multiword fields as arrays)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _flatten_regs(
    node: Dict[str, Any], path: List[str]
) -> Iterable[Tuple[List[str], int, int]]:
    """Yield (path, address, size_bytes) for every leaf register."""
    for key, value in node.items():
        if isinstance(value, dict) and {"address", "size"} <= set(value.keys()):
            yield path + [key], int(value["address"]), int(value["size"])
        elif isinstance(value, dict):
            yield from _flatten_regs(value, path + [key])
        else:
            raise TypeError(f"Unexpected leaf at {'.'.join(path+[key])}: {value!r}")


def _sanitize(parts: List[str], upper: bool) -> str:
    joined = "_".join(parts)
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", joined)
    return cleaned.upper() if upper else cleaned.lower()


def generate_header(json_path: Path, prefix: str) -> str:
    data = json.loads(json_path.read_text())
    regs_root = data["registers"]
    int(data["data_width"])
    int(data.get("granularity", 8))
    int(data["address_width"])

    items = list(_flatten_regs(regs_root, []))
    # sort by address for deterministic output
    items.sort(key=lambda x: x[1])

    # helpers
    def macro(parts):
        return f"{prefix}_{_sanitize(parts, upper=True)}"

    def field(parts):
        return _sanitize(parts, upper=False)

    lines: List[str] = []
    guard = f"{prefix}_H"
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")

    # enum
    lines.append("typedef enum {")
    for parts, addr, _ in items:
        lines.append(f"    {macro(parts)} = 0x{addr:04X}u,")
    lines.append(f"}} {prefix.lower()}_offsets_t;")
    lines.append("")

    lines.append("")
    lines.append(f"#endif /* {guard} */")

    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate C header from CSR JSON map")
    parser.add_argument(
        "--json", type=Path, required=True, help="Path to CSR JSON file"
    )
    parser.add_argument("--out", type=Path, required=True, help="Path to output header")
    parser.add_argument(
        "--prefix", default="PIXELFORGE_CSR", help="Macro/identifier prefix"
    )
    args = parser.parse_args(argv)

    header = generate_header(args.json, args.prefix)
    args.out.write_text(header)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
