"""Modal CLI for ambition_sfx_renderer.

Subcommands:

    render <cue>       Render a single cue YAML to output/<cue>/.
    render-all         Render every cue in sounds/active/ by default.
    audit [root]       Print loudness/peak stats for rendered audio files.
    list               List available cues.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ambition_sfx_renderer.audit import audit_output_tree, print_audit
from ambition_sfx_renderer.errors import SfxRenderError
from ambition_sfx_renderer.paths import output_root, sounds_root
from ambition_sfx_renderer.render import render_file
from ambition_sfx_renderer.schema import find_cue, iter_cue_files, load_cue


def cmd_render(args: argparse.Namespace) -> int:
    cue_path = find_cue(args.cue, root=args.sounds_root)
    if cue_path is None:
        print(f"error: cue not found: {args.cue}", file=sys.stderr)
        return 2
    out_root = args.outdir
    report = render_file(cue_path, out_root=out_root, write_wav=not args.no_wav, write_ogg=not args.no_ogg)
    print(
        f"rendered {report['id']}: duration={report['duration_seconds']:.3f}s "
        f"peak={report['peak_db']:.1f}dB rms={report['rms_db']:.1f}dB"
    )
    for kind, path in report.get("outputs", {}).items():
        print(f"  {kind}: {path}")
    return 0


def cmd_render_all(args: argparse.Namespace) -> int:
    cues = iter_cue_files(args.sounds_root, group=args.group)
    if not cues:
        print(f"error: no cue files found in {args.sounds_root / args.group}", file=sys.stderr)
        return 2
    failed: list[str] = []
    for cue_path in cues:
        try:
            report = render_file(
                cue_path,
                out_root=args.outdir,
                write_wav=not args.no_wav,
                write_ogg=not args.no_ogg,
            )
            print(
                f"rendered {report['id']}: duration={report['duration_seconds']:.3f}s "
                f"peak={report['peak_db']:.1f}dB rms={report['rms_db']:.1f}dB"
            )
        except Exception as ex:  # noqa: BLE001 - CLI should continue batch renders.
            failed.append(f"{cue_path.name}: {ex}")
            print(f"FAILED {cue_path}: {ex}", file=sys.stderr)
            if not args.keep_going:
                break
    if failed:
        print("Failures:", file=sys.stderr)
        for item in failed:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"OK: rendered {len(cues)} cue(s)")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    rows = audit_output_tree(args.root)
    print_audit(rows)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for path in iter_cue_files(args.sounds_root, group=args.group):
        try:
            spec = load_cue(path)
            print(f"{spec.cue_id:20s} {path}")
        except Exception as ex:  # noqa: BLE001
            print(f"INVALID {path}: {ex}", file=sys.stderr)
    return 0


def add_common_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sounds-root", type=Path, default=sounds_root(), help="cue search root")
    p.add_argument("--outdir", type=Path, default=output_root(), help="output root")
    p.add_argument("--no-wav", action="store_true", help="do not write WAV debug output")
    p.add_argument("--no-ogg", action="store_true", help="do not write OGG output")
    p.add_argument("--force", action="store_true", help="accepted for regen.sh compatibility; currently always renders")


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
    p_all.add_argument("--keep-going", action="store_true", default=True, help="continue after failed cue")
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
