"""Command-line interface for reproducible experiment presets."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__:
    from . import runner
else:  # Support direct invocation by absolute path from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments import runner


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(_jsonable(value), indent=2, sort_keys=True))


def _preset_argument(parser: argparse.ArgumentParser, *, default: str | None = None) -> None:
    parser.add_argument(
        "preset",
        nargs="?" if default is not None else None,
        default=default,
        help="built-in preset name or path to a YAML preset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="override the directory under which preset outputs are stored",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.run_experiments",
        description="List, run, inspect, validate, and audit reproducible experiment presets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list the six built-in presets and their current status")
    list_parser.add_argument("--output-root", type=Path, help="override the experiment output root")

    run_parser = subparsers.add_parser("run", help="run a preset")
    _preset_argument(run_parser)
    run_parser.add_argument("--slot", action="append", default=[], help="select a slot key; repeat to select more")
    run_parser.add_argument("--seed", action="append", type=int, default=[], help="select a root seed; repeat to select more")
    resume = run_parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True, help="reuse matching validated attempts (default)")
    resume.add_argument("--no-resume", dest="resume", action="store_false", help="always create new attempts")
    run_parser.add_argument("--keep-going", action="store_true", help="continue after an individual run fails")
    git_safety = run_parser.add_mutually_exclusive_group()
    git_safety.add_argument("--require-clean-git", action="store_true", help="refuse to run unless Git reports a clean worktree")
    git_safety.add_argument("--allow-dirty-git", action="store_true", help="explicitly allow a dirty or unavailable Git state (default behavior)")
    run_parser.add_argument("--confirm-final", action="store_true", help="explicitly authorize execution of a final preset")

    status_parser = subparsers.add_parser("status", help="show complete, unavailable, missing, and failed slot counts")
    _preset_argument(status_parser)

    validate_parser = subparsers.add_parser("validate", help="validate completed attempts and refresh status indexes")
    _preset_argument(validate_parser)

    audit_parser = subparsers.add_parser("audit-events", help="generate transition-event coverage witnesses")
    _preset_argument(audit_parser, default="smoke")

    return parser


def _cmd_list(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for name in runner.list_presets():
        row: dict[str, Any] = {"preset": name, "final": name in runner.FINAL_PRESETS}
        try:
            manifest = runner.status_preset(name, output_root=args.output_root)
            row.update(
                status="ok",
                planned=manifest.get("planned", 0),
                complete=manifest.get("complete", 0),
                unavailable=manifest.get("unavailable", 0),
                missing=manifest.get("missing", 0),
                failed=manifest.get("failed", 0),
                running_incomplete=manifest.get("running_incomplete", 0),
            )
        except Exception as exc:  # A broken preset must remain visible in list output.
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    _print_json(rows)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    summary = runner.run_preset(
        args.preset,
        output_root=args.output_root,
        slots=args.slot,
        seeds=args.seed,
        resume=args.resume,
        keep_going=args.keep_going,
        confirm_final=args.confirm_final,
        require_clean_git=args.require_clean_git,
    )
    _print_json(summary)
    return 1 if summary.failed else 0


def _cmd_status(args: argparse.Namespace) -> int:
    manifest = runner.status_preset(args.preset, output_root=args.output_root)
    _print_json(manifest)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    result = runner.validate_preset(args.preset, output_root=args.output_root)
    _print_json(result)
    return 1 if result.get("invalid_attempts", 0) else 0


def _cmd_audit_events(args: argparse.Namespace) -> int:
    witnesses = runner.audit_events(args.preset, output_root=args.output_root)
    covered = {witness.event for witness in witnesses}
    required = {event.value for event in runner.EventType} if hasattr(runner, "EventType") else covered
    _print_json({"preset": str(args.preset), "witnesses": witnesses, "missing": sorted(required - covered)})
    return 1 if required - covered else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "list": _cmd_list,
        "run": _cmd_run,
        "status": _cmd_status,
        "validate": _cmd_validate,
        "audit-events": _cmd_audit_events,
    }
    try:
        return commands[args.command](args)
    except PermissionError as exc:
        print(f"error: {exc}; pass --confirm-final only after reviewing the final preset", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
