"""Modal CLI for ambition_sfx_renderer.

Subcommands:

    render <cue>       Render a single cue YAML to output/<cue>/.
    render-all         Render every cue in sounds/active/ by default.
    audit [root]       Print loudness/peak stats for rendered audio files.
    list               List available cues.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
from pathlib import Path
from typing import Any

from ambition_sfx_renderer.audit import audit_output_tree, print_audit
from ambition_sfx_renderer.errors import SfxRenderError
from ambition_sfx_renderer.paths import output_root, sounds_root
from ambition_sfx_renderer.render import DEFAULT_WAV_MAX_SECONDS, render_file
from ambition_sfx_renderer.schema import find_cue, iter_cue_files, load_cue


def _render_one_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """ProcessPool worker for rendering one cue."""
    cue_path = Path(payload["cue_path"])
    report = render_file(
        cue_path,
        out_root=Path(payload["outdir"]),
        format_policy=payload["format_policy"],
        no_wav=payload["no_wav"],
        no_ogg=payload["no_ogg"],
        wav_max_seconds=payload["wav_max_seconds"],
        force=payload["force"],
    )
    return {"ok": True, "path": str(cue_path), "report": report}


def _parse_jobs(value: str) -> int:
    value = str(value).strip().lower()
    if value in {"auto", "0"}:
        return max(1, (os.cpu_count() or 1))
    jobs = int(value)
    if jobs < 1:
        raise argparse.ArgumentTypeError("--jobs must be a positive integer or 'auto'")
    return jobs


def _format_report(report: dict[str, Any]) -> str:
    verb = "skipped" if report.get("skipped") else "rendered"
    outputs = ",".join(sorted((report.get("outputs") or {}).keys())) or "manifest"
    policy = report.get("resolved_format_policy", report.get("format_policy", "?"))
    return (
        f"{verb} {report['id']}: duration={report['duration_seconds']:.3f}s "
        f"policy={policy} outputs={outputs} "
        f"peak={report['peak_db']:.1f}dB rms={report['rms_db']:.1f}dB"
    )


def cmd_render(args: argparse.Namespace) -> int:
    cue_path = find_cue(args.cue, root=args.sounds_root)
    if cue_path is None:
        print(f"error: cue not found: {args.cue}", file=sys.stderr)
        return 2
    report = render_file(
        cue_path,
        out_root=args.outdir,
        format_policy=args.format_policy,
        no_wav=args.no_wav,
        no_ogg=args.no_ogg,
        wav_max_seconds=args.wav_max_seconds,
        force=args.force,
    )
    print(_format_report(report))
    for kind, path in report.get("outputs", {}).items():
        print(f"  {kind}: {path}")
    return 0


def cmd_render_all(args: argparse.Namespace) -> int:
    cues = iter_cue_files(args.sounds_root, group=args.group)
    if not cues:
        print(f"error: no cue files found in {args.sounds_root / args.group}", file=sys.stderr)
        return 2
    jobs = _parse_jobs(args.jobs)
    payloads = [
        {
            "cue_path": str(cue_path),
            "outdir": str(args.outdir),
            "format_policy": args.format_policy,
            "no_wav": args.no_wav,
            "no_ogg": args.no_ogg,
            "wav_max_seconds": args.wav_max_seconds,
            "force": args.force,
        }
        for cue_path in cues
    ]
    failed: list[str] = []
    completed = 0
    skipped = 0
    rendered = 0
    print(
        f"render-all: {len(payloads)} cue(s), jobs={jobs}, force={args.force}, "
        f"format_policy={args.format_policy}, wav_max_seconds={args.wav_max_seconds:.3f}"
    )
    if jobs == 1:
        for payload in payloads:
            cue_path = Path(payload["cue_path"])
            try:
                result = _render_one_worker(payload)
                report = result["report"]
                completed += 1
                skipped += int(bool(report.get("skipped")))
                rendered += int(not bool(report.get("skipped")))
                print(_format_report(report))
            except Exception as ex:  # noqa: BLE001 - batch command should summarize failures.
                failed.append(f"{cue_path.name}: {ex}")
                print(f"FAILED {cue_path}: {ex}", file=sys.stderr)
                if args.fail_fast:
                    break
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_path = {
                executor.submit(_render_one_worker, payload): Path(payload["cue_path"])
                for payload in payloads
            }
            for future in cf.as_completed(future_to_path):
                cue_path = future_to_path[future]
                try:
                    result = future.result()
                    report = result["report"]
                    completed += 1
                    skipped += int(bool(report.get("skipped")))
                    rendered += int(not bool(report.get("skipped")))
                    print(_format_report(report))
                except Exception as ex:  # noqa: BLE001 - batch command should summarize failures.
                    failed.append(f"{cue_path.name}: {ex}")
                    print(f"FAILED {cue_path}: {ex}", file=sys.stderr)
                    if args.fail_fast:
                        executor.shutdown(cancel_futures=True)
                        break
    if failed:
        print("Failures:", file=sys.stderr)
        for item in failed:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"OK: {completed} cue(s); rendered={rendered}, skipped={skipped}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    rows = audit_output_tree(args.root)
    print_audit(rows)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for path in iter_cue_files(args.sounds_root, group=args.group):
        try:
            spec = load_cue(path)
            print(f"{spec.cue_id:40s} {spec.duration_seconds:6.3f}s {path}")
        except Exception as ex:  # noqa: BLE001
            print(f"INVALID {path}: {ex}", file=sys.stderr)
    return 0


def add_common_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sounds-root", type=Path, default=sounds_root(), help="cue search root")
    p.add_argument("--outdir", type=Path, default=output_root(), help="output root")
    p.add_argument(
        "--format-policy",
        choices=("auto", "both", "wav", "ogg"),
        default="auto",
        help=(
            "output format policy. default 'auto' writes WAV for cues <= "
            f"{DEFAULT_WAV_MAX_SECONDS:.3f}s and OGG for cues above that."
        ),
    )
    p.add_argument(
        "--wav-max-seconds",
        type=float,
        default=DEFAULT_WAV_MAX_SECONDS,
        help="duration boundary used by --format-policy auto; cues longer than this write OGG",
    )
    p.add_argument("--no-wav", action="store_true", help="disable WAV output after applying format policy")
    p.add_argument("--no-ogg", action="store_true", help="disable OGG output after applying format policy")
    p.add_argument("--force", action="store_true", help="force render even when output manifest is current")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ambition_sfx_renderer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="Render one cue")
    p_render.add_argument("cue", help="cue id or YAML path")
    add_common_render_args(p_render)
    p_render.set_defaults(func=cmd_render)

    p_all = sub.add_parser("render-all", help="Render all cues")
    p_all.add_argument("--group", default="active", help="sounds/<group>/ to render")
    p_all.add_argument(
        "--jobs",
        default="auto",
        help="number of parallel worker processes, or 'auto' (default)",
    )
    p_all.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop scheduling/processing after the first failed cue",
    )
    add_common_render_args(p_all)
    p_all.set_defaults(func=cmd_render_all)

    p_audit = sub.add_parser("audit", help="Audit rendered outputs")
    p_audit.add_argument("root", type=Path, nargs="?", default=output_root())
    p_audit.set_defaults(func=cmd_audit)

    p_list = sub.add_parser("list", help="List available cues")
    p_list.add_argument("--sounds-root", type=Path, default=sounds_root())
    p_list.add_argument("--group", default="active")
    p_list.set_defaults(func=cmd_list)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SfxRenderError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
