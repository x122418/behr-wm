#!/usr/bin/env python3
"""Audit whether paired TextWorld CR/CR-pw evaluation can be started."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXECUTABLE_GAME_EXTENSIONS = {".z8", ".ulx"}
GAME_ID_PATTERN = re.compile(r"^textworld_(\d+)$")


def audit_cr_readiness(
    agent_contexts: list[dict[str, Any]],
    wm_contexts: list[dict[str, Any]],
    game_files: list[str],
) -> dict[str, Any]:
    """Report necessary context pairing and executable-game prerequisites."""
    agent_ids = {row.get("data_idx") for row in agent_contexts}
    wm_ids = {row.get("id") for row in wm_contexts}
    agent_ids.discard(None)
    wm_ids.discard(None)
    paired_ids = agent_ids & wm_ids
    executable_game_files = [
        path
        for path in game_files
        if Path(path).suffix.lower() in EXECUTABLE_GAME_EXTENSIONS
    ]
    game_ids = set()
    for path in executable_game_files:
        match = GAME_ID_PATTERN.fullmatch(Path(path).stem)
        if match is not None:
            game_ids.add(int(match.group(1)))
    missing_game_ids = sorted(paired_ids - game_ids, key=str)
    blockers = []
    if agent_ids != wm_ids:
        blockers.append("agent/WM context IDs do not match")
    if not executable_game_files:
        blockers.append("no executable TextWorld game files")
    if len(executable_game_files) < len(paired_ids):
        blockers.append("fewer game files than paired evaluation contexts")
    if missing_game_ids:
        blockers.append("missing executable games for paired context IDs")
    return {
        "agent_context_count": len(agent_contexts),
        "wm_context_count": len(wm_contexts),
        "paired_context_count": len(paired_ids),
        "agent_only_ids": sorted(agent_ids - wm_ids, key=str),
        "wm_only_ids": sorted(wm_ids - agent_ids, key=str),
        "game_file_count": len(executable_game_files),
        "game_file_extensions": sorted(
            {Path(path).suffix for path in executable_game_files}
        ),
        "paired_game_count": len(paired_ids & game_ids),
        "missing_game_ids": missing_game_ids,
        "blocking_reasons": blockers,
        "ready": not blockers,
        "note": (
            "File count is only a necessary check; task-to-game ID mapping must "
            "still pass a reset/replay smoke test."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TextWorld CR/CR-pw inputs")
    parser.add_argument("--agent-contexts", type=Path, required=True)
    parser.add_argument("--wm-contexts", type=Path, required=True)
    parser.add_argument("--games-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    agent_contexts = json.loads(args.agent_contexts.read_text(encoding="utf-8"))
    wm_contexts = json.loads(args.wm_contexts.read_text(encoding="utf-8"))
    game_files = (
        sorted(str(path) for path in args.games_dir.iterdir() if path.is_file())
        if args.games_dir.is_dir()
        else []
    )
    report = audit_cr_readiness(agent_contexts, wm_contexts, game_files)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
