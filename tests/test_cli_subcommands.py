"""Tests for the new top-level ``prepare`` / ``generate`` / ``workspace``
subcommands on ``src.cli.build_parser``.

The existing ``run`` / ``cache`` / ``glossary`` subcommands must keep
working; these tests only cover the new dispatch surface required by
Task 12 of the workspace-architecture plan.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli import build_parser  # noqa: E402


def test_parser_has_prepare_generate_workspace() -> None:
    parser = build_parser()

    # 1) ``prepare`` subcommand: --input, --source-language, --target-language
    prepare_args = parser.parse_args(
        [
            "prepare",
            "--input",
            "clip.mp4",
            "--source-language",
            "en",
            "--target-language",
            "es",
        ]
    )
    assert prepare_args.command == "prepare"

    # 2) ``generate`` subcommand: optional positional <wid>
    generate_args = parser.parse_args(["generate", "my-clip-20260613-deadbeef"])
    assert generate_args.command == "generate"
    assert generate_args.workspace_id == "my-clip-20260613-deadbeef"

    # 3) ``workspace list`` subcommand (nested subsubparser)
    workspace_args = parser.parse_args(["workspace", "list"])
    assert workspace_args.command == "workspace"
    assert workspace_args.workspace_command == "list"
