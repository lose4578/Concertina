"""`concertina-data-quality`: scan a LeRobot v3 dataset, compile its quality package and write a cleaned copy.

    concertina-data-quality scan --dataset DATA --output RUN [--contract droid] [--config scan.yaml] [--workers 8]
    concertina-data-quality compile --run RUN [--cleaning-policy] [--decisions reviewed.json]
    concertina-data-quality materialize --package RUN/package --dataset DATA --output CLEAN [--media symlink]

The package can also be given to training (``DatasetSpec.quality_package``) to filter samples and mask action
targets of the source dataset without rewriting it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from concertina.data_quality.config import load_scan_config
from concertina.data_quality.materialize import MEDIA_MODES, materialize
from concertina.data_quality.pipeline import compile_package, scan_dataset
from concertina.data_quality.policy import PausePolicyConfig


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="concertina-data-quality", description=__doc__.split("\n\n")[0])
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="scan a dataset into <output>/scan (the dataset is only read)")
    scan.add_argument("--dataset", type=Path, required=True, help="local LeRobot v3 dataset directory")
    scan.add_argument("--output", type=Path, required=True, help="run directory")
    scan.add_argument("--contract", help="source contract (droid, agibot, robomind_*, calvin)")
    scan.add_argument("--config", type=Path, help="YAML overriding the contract's scan preset")
    scan.add_argument("--name", help="dataset name recorded in the package (default: directory name)")
    scan.add_argument("--workers", type=_positive_int, default=1)
    scan.add_argument("--max-files", type=_positive_int, help="scan only this many data files (partial scan)")
    scan.add_argument(
        "--require-ffmpeg",
        action="store_true",
        help="fail instead of warning when the camera stage cannot find the ffmpeg executable",
    )

    compile_ = commands.add_parser("compile", help="compile <run>/scan into a quality package")
    compile_.add_argument("--run", type=Path, required=True)
    compile_.add_argument("--output", type=Path, help="package directory (default: <run>/package)")
    compile_.add_argument("--decisions", type=Path, help="JSON list of reviewed decisions citing finding ids")
    compile_.add_argument(
        "--cleaning-policy", action="store_true", help="derive frame exclusions from the pause evidence"
    )
    compile_.add_argument("--minimum-exclusion-seconds", type=float, default=3.0)
    compile_.add_argument(
        "--allow-missing-visual-evidence",
        action="store_true",
        help="let pauses without camera evidence be excluded (by default they are kept)",
    )
    compile_.add_argument("--max-coverage-loss", type=float, default=0.10)
    compile_.add_argument("--dataset", type=Path, help="dataset root if it moved since the scan")

    apply = commands.add_parser("materialize", help="write the cleaned dataset described by a package")
    apply.add_argument("--package", type=Path, required=True)
    apply.add_argument("--dataset", type=Path, required=True, help="the scanned source dataset")
    apply.add_argument("--output", type=Path, required=True, help="new dataset directory (must not exist)")
    apply.add_argument("--media", choices=MEDIA_MODES, default="symlink", help="how video files are provided")
    apply.add_argument("--workers", type=_positive_int, default=1)
    apply.add_argument(
        "--rotation-stats-source",
        help="also write meta/rotation_stats.json for this Concertina source schema",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command == "scan":
        config = load_scan_config(args.config, contract=args.contract)
        if args.require_ffmpeg:
            config = dataclasses.replace(
                config, visual_motion=dataclasses.replace(config.visual_motion, require_ffmpeg=True)
            )
        result = scan_dataset(
            args.dataset,
            output=args.output,
            config=config,
            name=args.name,
            workers=args.workers,
            max_files=args.max_files,
        )
    elif args.command == "compile":
        decisions = []
        if args.decisions is not None:
            loaded = json.loads(args.decisions.read_text(encoding="utf-8"))
            decisions = loaded.get("decisions") if isinstance(loaded, dict) else loaded
            if not isinstance(decisions, list):
                raise SystemExit(f"{args.decisions} must hold a list of decisions")
        policy = (
            PausePolicyConfig(
                minimum_exclusion_seconds=args.minimum_exclusion_seconds,
                require_visual_evidence=not args.allow_missing_visual_evidence,
            )
            if args.cleaning_policy
            else None
        )
        package = compile_package(
            args.run,
            output=args.output,
            decisions=decisions,
            cleaning_policy=policy,
            max_coverage_loss=args.max_coverage_loss,
            source_root=args.dataset,
        )
        result = {
            "package": str(package.root),
            "release_ready": package.release_ready,
            "release_blockers": package.release_blockers,
            "scan_warnings": package.manifest["scan"]["warnings"],
            "coverage": package.manifest["coverage"],
            "decisions": package.manifest["decisions"],
        }
    else:
        result = materialize(
            args.package,
            source_root=args.dataset,
            output_root=args.output,
            media_mode=args.media,
            workers=args.workers,
            rotation_stats_source=args.rotation_stats_source,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
