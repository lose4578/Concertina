"""Scan a dataset and compile its quality package.

``scan_dataset`` writes the evidence of one LeRobot v3 dataset to ``<run>/scan`` (tables as JSONL plus
``scan.json``). ``compile_package`` turns it into a `concertina.data.quality.QualityPackage`: deterministic contract
quarantines, optional human decisions, an optional cleaning policy, per-task coverage and the release gates.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from concertina.data.quality import QualityPackage, read_jsonl, union_intervals
from concertina.data_quality import video, visual_motion
from concertina.data_quality.checks import json_safe
from concertina.data_quality.config import ScanConfig
from concertina.data_quality.policy import (
    PausePolicyConfig,
    compile_pause_policy,
    extract_auxiliary_motion,
    load_agibot_phases,
    normalize_intervals,
)
from concertina.data_quality.scan import (
    PAUSE_RULE,
    load_episode_records,
    read_episode_table,
    read_info,
    run_parallel,
    scan_data_file,
)

logger = logging.getLogger(__name__)

SCAN_DIR = "scan"
SCAN_SUMMARY = "scan.json"
SCAN_TABLES = ("episode_metrics", "interval_findings", "episode_decisions", "field_masks", "video_findings")
HUMAN_ACTIONS = (
    "keep",
    "reject_finding",
    "quarantine_episode",
    "exclude_frame_interval",
    "mask_field_interval",
)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def scan_dataset(
    root: str | Path,
    *,
    output: str | Path,
    config: ScanConfig,
    name: str | None = None,
    workers: int = 1,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Scan ``root`` into ``<output>/scan``; ``max_files`` limits the data files (a partial scan)."""
    root = Path(root).expanduser().resolve()
    name = name or root.name
    info = read_info(root)
    episodes = load_episode_records(root, info)
    by_file: dict[str, dict[int, Any]] = defaultdict(dict)
    for record in episodes.values():
        by_file[record.data_path][record.episode_index] = record
    missing_files = sorted(path for path in by_file if not (root / path).is_file())
    present = [path for path in sorted(by_file) if path not in missing_files]
    selected = present if max_files is None else present[:max_files]
    logger.info(
        "%s: scanning %d of %d data files with %d workers", name, len(selected), len(by_file), workers
    )
    results = run_parallel(
        scan_data_file,
        [(root, path, by_file[path], config, name) for path in selected],
        workers,
    )
    tables: dict[str, list[dict[str, Any]]] = {table: [] for table in SCAN_TABLES}
    scanned: set[int] = set()
    missing_columns = set()
    for result in results:
        for table in ("episode_metrics", "interval_findings", "episode_decisions", "field_masks"):
            tables[table].extend(result[table])
        scanned.update(result["episode_indices"])
        missing_columns.update(result["missing_required_columns"])
    expected = {episode for path in selected for episode in by_file[path]}
    unscanned = sorted(expected - scanned)
    duplicates = sorted(
        e for e, n in Counter(r["episode_index"] for r in tables["episode_metrics"]).items() if n > 1
    )

    video_summary: dict[str, Any] = {"enabled": config.video.enabled}
    if config.video.enabled:
        rows = read_episode_table(root)
        referenced = video.referenced_video_files(rows, info)
        missing_videos = [item for item in referenced if not (root / item.relative_path).is_file()]
        sampled = video.sample_video_files(
            [item for item in referenced if item not in missing_videos], config.video.sample_fraction
        )
        expectations = video.camera_expectations(info)
        probes = run_parallel(
            video.probe_and_judge,
            [(root, item, config.video, expectations.get(item.camera)) for item in sampled],
            workers,
        )
        tables["video_findings"] = [
            *(
                {
                    "camera": item.camera,
                    "source_path": item.relative_path,
                    "rule_id": "VID.MISSING_FILE",
                    "severity": "critical",
                    "message": "Episode metadata references a video file that does not exist",
                }
                for item in missing_videos
            ),
            *(finding for probe in probes for finding in probe["findings"]),
        ]
        video_summary.update(
            {
                "referenced_files": len(referenced),
                "missing_files": len(missing_videos),
                "probed_files": len(sampled),
                "findings_by_rule": dict(Counter(row["rule_id"] for row in tables["video_findings"])),
            }
        )

    warnings: list[str] = []
    motion_summary: dict[str, Any] = {"enabled": config.visual_motion.enabled}
    if config.visual_motion.enabled:
        if shutil.which(config.visual_motion.ffmpeg_binary) is None:
            message = (
                f"FFmpeg executable {config.visual_motion.ffmpeg_binary!r} is not on PATH: pause candidates get no "
                "camera evidence, so the cleaning policy keeps every pause. Install FFmpeg (or set "
                "visual_motion.ffmpeg_binary); visual_motion.require_ffmpeg / --require-ffmpeg makes this an error."
            )
            if config.visual_motion.require_ffmpeg:
                raise RuntimeError(message)
            logger.warning(message)
            warnings.append(message)
        contract = config.source_contract
        candidates, skipped = visual_motion.select_candidates(
            tables["interval_findings"],
            canonical_action_column=contract.canonical_action_column,
            minimum_seconds=config.visual_motion.minimum_candidate_seconds,
        )
        cameras = [camera for camera in contract.video_keys if camera in info["features"]]
        rows = {int(row["episode_index"]): row for row in read_episode_table(root)}
        groups = visual_motion.plan_groups(candidates, episode_rows=rows, info=info, cameras=cameras)
        results = run_parallel(
            visual_motion.analyze_group,
            [(root, group) for group in groups],
            workers,
            kwargs={"fps": float(info["fps"]), "config": config.visual_motion},
        )
        evidence = visual_motion.aggregate_evidence(
            candidates, results, cameras=cameras, config=config.visual_motion
        )
        evidence.update(visual_motion.skipped_evidence(skipped))
        for finding in tables["interval_findings"]:
            if finding["rule_id"] == PAUSE_RULE and finding["finding_id"] in evidence:
                finding["visual_motion_evidence"] = evidence[finding["finding_id"]]
        motion_summary.update(
            {
                "candidates": len(candidates),
                "skipped": len(skipped),
                "groups": len(groups),
                "dispositions": dict(Counter(item["disposition"] for item in evidence.values())),
                "tools": visual_motion.tool_versions(config.visual_motion),
            }
        )

    scan_dir = Path(output) / SCAN_DIR
    scan_dir.mkdir(parents=True, exist_ok=True)
    for table, rows in tables.items():
        _write_jsonl(scan_dir / f"{table}.jsonl", [json_safe(row) for row in rows])
    summary = {
        "dataset": {
            "name": name,
            "contract": config.contract,
            "codebase_version": info["codebase_version"],
            "fps": float(info["fps"]),
            "total_episodes": int(info["total_episodes"]),
            "total_frames": int(info["total_frames"]),
        },
        "source_root": str(root),
        "config": config.to_dict(),
        "scope": "full" if len(selected) == len(by_file) else "partial",
        "data_files": len(by_file),
        "scanned_data_files": len(selected),
        "missing_data_files": missing_files,
        "missing_required_columns": sorted(missing_columns),
        "unscanned_catalog_episodes": unscanned[:100],
        "duplicate_scanned_episodes": duplicates[:100],
        "episodes": len(tables["episode_metrics"]),
        "findings_by_rule": dict(Counter(row["rule_id"] for row in tables["interval_findings"])),
        "quarantined_episodes": sum(row["decision"] == "quarantine" for row in tables["episode_decisions"]),
        "video": video_summary,
        "visual_motion": motion_summary,
        "warnings": warnings,
    }
    (scan_dir / SCAN_SUMMARY).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


# ── compile ─────────────────────────────────────────────────────────────


def load_scan(run: str | Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    scan_dir = Path(run) / SCAN_DIR
    summary = json.loads((scan_dir / SCAN_SUMMARY).read_text(encoding="utf-8"))
    return summary, {table: read_jsonl(scan_dir / f"{table}.jsonl") for table in SCAN_TABLES}


def _contained(start: int, end: int, span: Sequence[int]) -> bool:
    return span[0] <= start < end <= span[1]


def apply_human_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    findings: Mapping[str, Mapping[str, Any]],
    lengths: Mapping[int, int],
    episode_decisions: dict[int, dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
) -> Counter:
    """Apply reviewed decisions; each cites a finding (``candidate_id``) and stays inside its evidence.

    Actions: ``keep``/``reject_finding`` (recorded, no change), ``quarantine_episode``,
    ``exclude_frame_interval`` (``start_frame``/``end_frame`` within the finding, not overlapping
    camera-protected motion) and ``mask_field_interval`` (one exact source ``field`` of the finding, within
    its interval for that field).
    """
    counts: Counter = Counter()
    seen: set[str] = set()
    for number, decision in enumerate(decisions):
        where = f"human decision {number}"
        candidate_id, action = decision.get("candidate_id"), decision.get("action")
        if candidate_id not in findings:
            raise ValueError(f"{where}: unknown candidate_id {candidate_id!r}")
        if candidate_id in seen:
            raise ValueError(f"{where}: candidate {candidate_id!r} is decided twice")
        seen.add(candidate_id)
        if action not in HUMAN_ACTIONS:
            raise ValueError(f"{where}: action must be one of {HUMAN_ACTIONS}, got {action!r}")
        if not decision.get("reason"):
            raise ValueError(f"{where}: a reason is required")
        counts[action] += 1
        if action in ("keep", "reject_finding"):
            continue
        finding = findings[candidate_id]
        episode = int(finding["episode_index"])
        if decision.get("episode_index", episode) != episode:
            raise ValueError(f"{where}: episode_index disagrees with the finding")
        provenance = {
            "origin": "human",
            "candidate_id": candidate_id,
            "reason": str(decision["reason"]),
            **({"reviewer": str(decision["reviewer"])} if decision.get("reviewer") else {}),
        }
        if action == "quarantine_episode":
            row = episode_decisions[episode]
            row.update(
                {"decision": "quarantine", "reasons": sorted({*row["reasons"], "human_source_decision"})}
            )
            row.setdefault("human", []).append(provenance)
            continue
        start, end = int(decision["start_frame"]), int(decision["end_frame"])
        if not 0 <= start < end <= lengths[episode]:
            raise ValueError(f"{where}: [{start}, {end}) is outside episode {episode}")
        common = {
            "episode_index": episode,
            "start_frame": start,
            "end_frame": end,
            "rules": [finding["rule_id"]],
        }
        if action == "exclude_frame_interval":
            if finding.get("start_frame") is None or not _contained(
                start, end, (finding["start_frame"], finding["end_frame"])
            ):
                raise ValueError(f"{where}: the interval must lie inside the finding")
            evidence = finding.get("visual_motion_evidence") or {}
            for span in evidence.get("protected_motion_intervals", ()):
                if span["start_frame"] < end and span["end_frame"] > start:
                    raise ValueError(f"{where}: the interval overlaps camera-protected motion {span}")
            frame_rows.append({**common, "decision": "exclude_source_frames", **provenance})
        else:
            field = decision.get("field")
            metrics = finding.get("metrics") or {}
            pair = metrics.get("source_pair") or {}
            spans = pair.get("field_intervals") or metrics.get("field_intervals") or {}
            span = spans.get(field)
            if span is None:
                raise ValueError(
                    f"{where}: field {field!r} has no evidence in the finding (has {sorted(spans)})"
                )
            if not _contained(start, end, (span["start_index"], span["end_index"])):
                raise ValueError(
                    f"{where}: the interval must lie inside the finding's interval for {field!r}"
                )
            mask_rows.append({**common, "field": field, "decision": "mask_field", **provenance})
    return counts


def _excluded_frames(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[tuple[int, int]]]:
    spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        spans[int(row["episode_index"])].append((int(row["start_frame"]), int(row["end_frame"])))
    return {episode: union_intervals(values) for episode, values in spans.items()}


def _retained(lengths, episode_decisions, frame_rows) -> dict[int, int]:
    excluded = _excluded_frames(frame_rows)
    return {
        episode: 0
        if episode_decisions[episode]["decision"] == "quarantine"
        else length - sum(e - s for s, e in excluded.get(episode, ()))
        for episode, length in lengths.items()
    }


def compile_package(
    run: str | Path,
    *,
    output: str | Path | None = None,
    decisions: Sequence[Mapping[str, Any]] = (),
    cleaning_policy: PausePolicyConfig | None = None,
    max_coverage_loss: float = 0.10,
    source_root: str | Path | None = None,
) -> QualityPackage:
    """Quality package of a scan (``<run>/package`` by default).

    ``cleaning_policy`` derives frame exclusions from the pause evidence (it reads the dataset at
    ``source_root``, by default the scanned root). ``max_coverage_loss`` bounds the fraction of frames removed
    by contract quarantines and human decisions; the policy's own removals are reported, not gated.
    """
    summary, tables = load_scan(run)
    dataset = summary["dataset"]
    lengths = {int(row["episode_index"]): int(row["frame_count"]) for row in tables["episode_metrics"]}
    episode_decisions = {int(row["episode_index"]): dict(row) for row in tables["episode_decisions"]}
    frame_rows: list[dict[str, Any]] = []
    mask_rows = [dict(row) for row in tables["field_masks"]]
    findings = {row["finding_id"]: row for row in tables["interval_findings"]}
    human = apply_human_decisions(
        decisions,
        findings=findings,
        lengths=lengths,
        episode_decisions=episode_decisions,
        frame_rows=frame_rows,
        mask_rows=mask_rows,
    )

    blockers: list[str] = []
    if summary["scope"] != "full":
        blockers.append(f"partial_scan:{summary['scanned_data_files']}/{summary['data_files']}_data_files")
    for key in ("missing_data_files", "missing_required_columns", "unscanned_catalog_episodes"):
        if summary[key]:
            blockers.append(f"{key}:{summary[key][:5]}")
    if summary["duplicate_scanned_episodes"]:
        blockers.append(f"duplicate_scanned_episodes:{summary['duplicate_scanned_episodes'][:5]}")
    scan_config = summary["config"]
    if scan_config["video"]["enabled"] and scan_config["video"]["required_for_release"]:
        blocking = Counter(
            row["rule_id"] for row in tables["video_findings"] if row["severity"] in video.BLOCKING_SEVERITIES
        )
        if blocking:
            blockers.append(f"video_findings:{dict(sorted(blocking.items()))}")

    source_frames = sum(lengths.values())
    pre_policy_retained = sum(_retained(lengths, episode_decisions, frame_rows).values())
    loss = 1.0 - pre_policy_retained / source_frames if source_frames else 1.0
    if loss > max_coverage_loss:
        blockers.append(f"coverage_loss:{loss:.4f}>{max_coverage_loss}")

    policy_rows: list[dict[str, Any]] = []
    policy_summary: dict[str, Any] | None = None
    scan_warnings = list(summary.get("warnings", []))
    for message in scan_warnings:
        logger.warning("%s: scan warning: %s", dataset["name"], message)
    if cleaning_policy is not None:
        root = Path(source_root or summary["source_root"])
        fps = float(dataset["fps"])
        contract = dataset["contract"]
        result = compile_pause_policy(
            contract_name=contract,
            fps=fps,
            findings=tables["interval_findings"],
            episode_lengths=lengths,
            auxiliary_motion=extract_auxiliary_motion(
                root=root,
                contract_name=contract,
                fps=fps,
                findings=tables["interval_findings"],
                config=cleaning_policy,
            ),
            phases_by_episode=load_agibot_phases(root) if contract == "agibot" else None,
            config=cleaning_policy,
        )
        policy_rows = list(result.decisions)
        for decision in result.applied():
            episode = decision["episode_index"]
            if decision["action"] == "quarantine_episode":
                row = episode_decisions[episode]
                row.update({"decision": "quarantine", "reasons": sorted({*row["reasons"], decision["rule"]})})
            elif decision["action"] == "exclude_frame_interval":
                frame_rows.append(
                    {
                        "episode_index": episode,
                        "start_frame": decision["start_frame"],
                        "end_frame": decision["end_frame"],
                        "decision": "exclude_source_frames",
                        "rules": [decision["rule"]],
                        "origin": "policy",
                        "policy_decision_id": decision["decision_id"],
                    }
                )
        blockers.extend(result.blockers)
        if result.unresolved_review_count:
            blockers.append(f"policy_unresolved_review:{result.unresolved_review_count}")
        policy_summary = {
            "policy_id": cleaning_policy.policy_id,
            "config": dataclasses.asdict(cleaning_policy),
            "decisions": dict(Counter(f"{row['status']}:{row['action']}" for row in policy_rows)),
        }

    # An episode whose every frame is excluded is quarantined instead.
    excluded = _excluded_frames(frame_rows)
    for episode, spans in excluded.items():
        if episode_decisions[episode]["decision"] == "keep" and spans == [(0, lengths[episode])]:
            row = episode_decisions[episode]
            row.update(
                {"decision": "quarantine", "reasons": sorted({*row["reasons"], "all_frames_excluded"})}
            )
    quarantined = {e for e, row in episode_decisions.items() if row["decision"] == "quarantine"}
    frame_rows = [row for row in frame_rows if row["episode_index"] not in quarantined]
    mask_rows = [row for row in mask_rows if row["episode_index"] not in quarantined]
    retained = _retained(lengths, episode_decisions, frame_rows)

    masked: Counter = Counter()
    for (episode, _field), spans in _group_masks(mask_rows).items():
        masked[episode] += sum(e - s for s, e in spans)
    metrics = []
    for row in tables["episode_metrics"]:
        episode = int(row["episode_index"])
        metrics.append(
            {
                **row,
                "decision": episode_decisions[episode]["decision"],
                "retained_frame_count": retained[episode],
                "field_masked_row_count": masked[episode],
            }
        )
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in metrics:
        key = (str(row.get("semantic_task")), str(row.get("embodiment_name")))
        groups[key].update(
            {
                "episode_count": 1,
                "frame_count": row["frame_count"],
                "retained_episode_count": int(row["decision"] == "keep"),
                "retained_frame_count": row["retained_frame_count"],
                "field_masked_row_count": row["field_masked_row_count"],
            }
        )
    distribution = [
        {"semantic_task": task, "embodiment_name": embodiment, **dict(counts)}
        for (task, embodiment), counts in sorted(groups.items())
    ]
    retained_frames = sum(retained.values())
    package_tables = {
        "episode_metrics": metrics,
        "interval_findings": tables["interval_findings"],
        "episode_decisions": [episode_decisions[e] for e in sorted(episode_decisions)],
        "frame_decisions": sorted(
            frame_rows, key=lambda r: (r["episode_index"], r["start_frame"], r["end_frame"])
        ),
        "field_masks": sorted(
            mask_rows, key=lambda r: (r["episode_index"], r["field"], r["start_frame"], r["end_frame"])
        ),
        "group_distribution": distribution,
        "video_findings": tables["video_findings"],
        **({"policy_decisions": policy_rows} if cleaning_policy is not None else {}),
    }
    package = QualityPackage.write(
        Path(output) if output is not None else Path(run) / "package",
        dataset=dataset,
        tables=package_tables,
        release_blockers=blockers,
        scan={
            **{
                key: summary[key]
                for key in ("scope", "data_files", "scanned_data_files", "video", "visual_motion")
            },
            "warnings": scan_warnings,
        },
        scan_config=scan_config,
        policy=policy_summary,
        coverage={
            "source_frames": source_frames,
            "retained_frames": retained_frames,
            "loss_fraction_before_policy": loss,
            "loss_fraction": 1.0 - retained_frames / source_frames if source_frames else 1.0,
            "max_loss_before_policy": max_coverage_loss,
        },
        decisions={
            "quarantined_episodes": len(quarantined),
            "excluded_frames": sum(
                e - s for spans in _excluded_frames(frame_rows).values() for s, e in spans
            ),
            "masked_rows": sum(masked.values()),
            "human": dict(human),
        },
    )
    logger.info(
        "%s: package %s, release_ready=%s, blockers=%s",
        dataset["name"],
        package.root,
        package.release_ready,
        package.release_blockers,
    )
    return package


def _group_masks(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], list[tuple[int, int]]]:
    spans: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        spans[(int(row["episode_index"]), str(row["field"]))].append((row["start_frame"], row["end_frame"]))
    return {
        key: normalize_intervals(values, start=0, end=max(e for _, e in values))
        for key, values in spans.items()
    }
