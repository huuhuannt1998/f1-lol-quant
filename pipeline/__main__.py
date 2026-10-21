"""Command line entry point.

    python -m pipeline update          fetch everything, forecast, write data/*.json
    python -m pipeline update --sport lol
    python -m pipeline update --offline   reuse cached data (no network)
    python -m pipeline demo            synthetic data end to end (no network)
    python -m pipeline backtest lol    tier-2 / international-K experiments on cached data
"""
from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path

# Known-benign warnings, matched by exact message so anything new still shows.
# * macOS: NumPy 2.0 with Apple's Accelerate BLAS raises spurious floating-point
#   flags in matrix products. All three ("divide by zero", "overflow", "invalid
#   value") always fire together, and a matrix product cannot divide by zero.
#   Each run prints a finite-feature check, so real NaN/inf would still show.
# * urllib3 v2 warns once at import on macOS's system Python (LibreSSL).
if sys.platform == "darwin":
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message=r"(divide by zero|overflow|invalid value) encountered in matmul")
warnings.filterwarnings("ignore", message=r"urllib3 v2 only supports OpenSSL")

from . import config, site  # noqa: E402 - filters must be set before imports


def _update(sports, fetch: bool, out_dir: Path | None = None) -> int:
    status, failed = {}, 0
    for sport in sports:
        try:
            if sport == "lol":
                from .lol.run import run as run_sport
            else:
                from .f1.run import run as run_sport
            payload = run_sport(fetch=fetch)
            site.write(sport, payload, out_dir)
            status[sport] = "ok"
        except Exception as e:  # noqa: BLE001 - one sport failing must not block the other
            traceback.print_exc()
            status[sport] = f"error: {type(e).__name__}: {e}"[:300]
            failed += 1
    site.write_meta(status, out_dir)
    print("status:", status)
    return 1 if failed == len(sports) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update")
    u.add_argument("--sport", choices=["lol", "f1", "all"], default="all")
    u.add_argument("--offline", action="store_true", help="no network; use cached data")
    d = sub.add_parser("demo")
    d.add_argument("--out", type=Path, default=config.ROOT / "cache" / "demo_site" / "data")
    b = sub.add_parser("backtest", help="offline experiments on cached data (no network)")
    b.add_argument("sport", choices=["lol"])
    args = ap.parse_args(argv)

    if args.cmd == "backtest":
        from .lol.experiments import compare
        compare()
        return 0

    if args.cmd == "update":
        sports = ["lol", "f1"] if args.sport == "all" else [args.sport]
        return _update(sports, fetch=not args.offline)

    if args.cmd == "demo":
        from . import synthetic
        synthetic.install()
        return _update(["lol", "f1"], fetch=False, out_dir=args.out)
    return 2


if __name__ == "__main__":
    sys.exit(main())
