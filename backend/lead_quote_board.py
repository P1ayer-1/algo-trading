"""Every lead-quote run at once, or its paper fills as they happen. READ ONLY.

    python backend\\lead_quote_board.py                          # one table: newest run per pair, then all runs pooled
    python backend\\lead_quote_board.py --watch 60               # ...redrawn every minute
    python backend\\lead_quote_board.py --since 2026-09-13T13:30 # only runs started since (UTC; or 12h, 2d)
    python backend\\lead_quote_board.py --follow                 # print each paper fill and exit as it lands
    python backend\\lead_quote_board.py --follow --intents       # ...and every post and cancel

`run_lead_quote.py --summary` reads one run; with a dozen pairs running that
is a dozen reports and no total. This reads every `data/<INST>/lead_quote/`
log and pools closed paper fills per pair and across pairs - the unit the S3
decision is taken in (STRATEGIES.md).

`--follow` tails the JSONL logs, which the runner flushes row by row, not the
`lq_*.out` console files, which Python block-buffers when redirected. It first
reads every log once, silently, so its running totals start from everything
already on disk, then prints the last `--history` lines and waits.

Reads files only. No keys, no network, no path to an order.
"""

from __future__ import annotations

import argparse
import calendar
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.layout import instrument_dirs  # noqa: E402
from trading.strategies.lead_quote.monitor import (  # noqa: E402
    LogTail, Narrator, board, run_started_ms, summarise)

COLOURS = {"win": "\033[32m", "loss": "\033[31m", "fill": "\033[36m", "warn": "\033[33m", "info": "\033[2m"}
RESET = "\033[0m"


def parse_since(text: Optional[str], now_ms: int) -> Optional[int]:
    if not text:
        return None
    text = text.strip()
    units = {"h": 3_600_000, "d": 86_400_000}
    if text[-1:] in units:
        try:
            return now_ms - int(float(text[:-1]) * units[text[-1]])
        except ValueError:
            pass
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return calendar.timegm(time.strptime(text, fmt)) * 1000
        except ValueError:
            continue
    raise SystemExit("--since {!r}: give a UTC time as 2026-09-13, 2026-09-13T13 or 2026-09-13T13:30, "
                     "or an age as 12h or 2d.".format(text))


def run_logs(data_dir: Path, instruments: Optional[List[str]], since_ms: Optional[int]) -> List[Path]:
    paths = []
    for inst_id, directory in instrument_dirs(data_dir).items():
        if instruments and inst_id not in instruments:
            continue
        for path in sorted((directory / "lead_quote").glob("*.jsonl")):
            started = run_started_ms(path)
            if since_ms is not None and started is not None and started < since_ms:
                continue
            paths.append(path)
    return paths


def show_board(args, instruments) -> None:
    now = int(time.time() * 1000)
    paths = run_logs(args.data_dir, instruments, parse_since(args.since, now))
    if not paths:
        raise SystemExit("no lead-quote run logs under {}/<INST>/lead_quote/{}.".format(
            args.data_dir, " for " + ",".join(instruments) if instruments else ""))
    for line in board([summarise(p) for p in paths], now):
        print(line)


def follow(args, instruments, colour: bool) -> None:
    started = int(time.time() * 1000)
    since_ms = parse_since(args.since, started)
    tail = LogTail(lambda: run_logs(args.data_dir, instruments, since_ms))
    narrator = Narrator(show_intents=args.intents)
    history = []
    for path, row in tail.poll():
        history += narrator.feed(path, row)
    narrator.check_silence(started)          # runs already silent are the board's news, not this feed's

    def emit(level: str, text: str) -> None:
        if colour and level in COLOURS:
            text = COLOURS[level] + text + RESET
        if args.bell and level in ("win", "loss"):
            text += "\a"
        print(text, flush=True)

    for level, text in history[-args.history:] if args.history else []:
        emit(level, text)
    print("following {} run logs under {} ({} live); so far {}. Ctrl-C to stop.".format(
        len(narrator.runs), args.data_dir,
        sum(1 for r in narrator.runs.values() if not (r.stopped or r.silent)), narrator.tally()), flush=True)
    try:
        while True:
            time.sleep(args.poll)
            for path, row in tail.poll():
                for level, text in narrator.feed(path, row):
                    emit(level, text)
            for level, text in narrator.check_silence(int(time.time() * 1000)):
                emit(level, text)
    except KeyboardInterrupt:
        print("\n" + narrator.tally())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--instruments", help="comma-separated instIds; default every pair with a lead_quote log")
    parser.add_argument("--since", help="only runs started at or after this UTC time (2026-09-13T13:30) or age (12h)")
    parser.add_argument("--watch", type=float, default=0.0, metavar="SECONDS", help="redraw the board every SECONDS")
    parser.add_argument("--follow", action="store_true", help="print paper fills and exits as they are written")
    parser.add_argument("--intents", action="store_true", help="with --follow: also every post and cancel")
    parser.add_argument("--history", type=int, default=10, help="with --follow: lines of past events to show first")
    parser.add_argument("--poll", type=float, default=1.0, help="with --follow: seconds between reads")
    parser.add_argument("--bell", action="store_true", help="with --follow: ring the terminal bell on each exit")
    parser.add_argument("--no-colour", "--no-color", dest="no_colour", action="store_true")
    args = parser.parse_args(argv)
    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()] if args.instruments else None
    colour = sys.stdout.isatty() and not args.no_colour

    if args.follow:
        follow(args, instruments, colour)
        return 0
    if not args.watch:
        show_board(args, instruments)
        return 0
    try:
        while True:
            if colour:
                print("\033[H\033[2J", end="")
            show_board(args, instruments)
            print("\nredrawn every {:g} s; Ctrl-C to stop".format(args.watch), flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
