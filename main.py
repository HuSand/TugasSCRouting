"""
Surabaya Public Facility Routing Platform
==========================================
Single entry point for all pipeline steps.

Usage
-----
    python main.py extract          # download OSM data  (~5-10 min, needs internet)
    python main.py explore          # profile + visualize data
    python main.py demo             # baseline routing demonstrations
    python main.py compare          # algorithm comparison benchmark
    python main.py all              # run everything in order

Options
-------
    --skip-existing                 # skip steps whose output files already exist
    --no-log-file                   # print logs to console only (no file)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make src importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

def setup_logging(cfg) -> logging.Logger:
    from datetime import datetime
    handlers = [logging.StreamHandler(sys.stdout)]
    if cfg.LOG_TO_FILE:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(cfg.LOG_DIR / f"run_{ts}.log"))
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("platform")


# ──────────────────────────────────────────────────────────────
# Step runners
# ──────────────────────────────────────────────────────────────

def run_step(name: str, fn, cfg, log: logging.Logger):
    log.info("")
    log.info("=" * 60)
    log.info(f"STEP: {name.upper()}")
    log.info("=" * 60)
    t0 = time.perf_counter()
    fn(cfg)
    elapsed = time.perf_counter() - t0
    log.info(f"[{name}] finished in {elapsed:.1f}s")


def _extract(cfg):
    from src.extract import run_extraction
    run_extraction(cfg)


def _explore(cfg):
    from src.explore import run_exploration
    run_exploration(cfg)


def _demo(cfg):
    from src.routing.demos import run_demos
    run_demos(cfg)


def _compare(cfg):
    from src.routing.benchmark import run_platform
    run_platform(cfg)


STEPS = {
    "extract": ("Extract facilities + road network from OSM", _extract),
    "explore": ("Profile and visualize extracted data",       _explore),
    "demo":    ("Run baseline routing demonstrations",        _demo),
    "compare": ("Run algorithm comparison benchmark",         _compare),
}


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    desc_lines = ["Available commands:"]
    for cmd, (desc, _) in STEPS.items():
        desc_lines.append(f"  {cmd:<10}  {desc}")
    desc_lines.append(f"  {'all':<10}  Run full pipeline (extract → explore → demo → compare)")

    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Surabaya Public Facility Routing Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(desc_lines),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=[*STEPS.keys(), "all"],
        metavar="command",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable writing logs to file",
    )
    parser.add_argument(
        "--parallel-legs",
        action="store_true",
        help="Enable leg-level parallelism in benchmark (more CPU cores, higher RAM usage)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    from settings import Settings
    cfg = Settings()
    if args.no_log_file:
        cfg.LOG_TO_FILE = False
    cfg.PARALLEL_LEGS = args.parallel_legs

    log = setup_logging(cfg)
    log.info("Surabaya Public Facility Routing Platform")
    log.info(f"Place : {cfg.PLACE}")
    log.info(f"Data  : {cfg.DATA_DIR}")
    if cfg.LOG_TO_FILE:
        log.info(f"Logs  : {cfg.LOG_DIR}")

    to_run = list(STEPS.items()) if args.command == "all" else [(args.command, STEPS[args.command])]

    total_start = time.perf_counter()
    for cmd, (_, fn) in to_run:
        run_step(cmd, fn, cfg, log)

    total = time.perf_counter() - total_start
    log.info("")
    log.info(f"All done in {total:.1f}s")


if __name__ == "__main__":
    main()