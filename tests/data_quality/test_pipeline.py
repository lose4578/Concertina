"""End-to-end data quality: scan, package, pause-cleaning policy, cleaned copy and training-time binding."""

import dataclasses
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from concertina.data.mixture import validate_dense_layout
from concertina.data.quality import PACKAGE_TABLES, QualityConfig, QualityPackage, load_quality_index
from concertina.data_quality import cli, video
from concertina.data_quality.config import ScanConfig, VideoConfig, VisualMotionConfig, load_scan_config
from concertina.data_quality.materialize import materialize, plan_segments
from concertina.data_quality.pipeline import compile_package, load_scan, scan_dataset
from concertina.data_quality.policy import PausePolicyConfig

DROID_CAMERAS = (
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
)


# The camera-motion evidence that lets the cleaning policy exclude pauses needs the FFmpeg executable.
requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="the camera-motion stage needs the ffmpeg executable on PATH"
)


def _frames(intervals) -> set[int]:
    return {frame for start, end in intervals for frame in range(start, end)}


@pytest.fixture(scope="module")
def droid_run(droid_like, tmp_path_factory) -> Path:
    run = tmp_path_factory.mktemp("droid_run")
    scan_dataset(droid_like, output=run, config=ScanConfig.preset("droid"), name="droid_like")
    return run


@pytest.fixture(scope="module")
def droid_package(droid_run) -> QualityPackage:
    return compile_package(droid_run, output=droid_run / "cleaned", cleaning_policy=PausePolicyConfig())


@requires_ffmpeg
def test_scan_reports_pauses_with_camera_evidence(droid_run, synthetic):
    summary, tables = load_scan(droid_run)
    assert summary["scope"] == "full" and summary["episodes"] == len(synthetic.DROID_EPISODES)
    assert summary["video"]["probed_files"] == 3 and not tables["video_findings"]
    pauses = [row for row in tables["interval_findings"] if row["rule_id"] == "TRJ.NEAR_STATIC_RUN"]
    assert {row["episode_index"] for row in pauses} == {0, 2}
    for row in pauses:
        static = synthetic.DROID_EPISODES[row["episode_index"]][1]
        # The first static frame repeats the frame before it.
        assert (row["start_frame"], row["end_frame"]) == (static[0] - 1, static[1])
        evidence = row["visual_motion_evidence"]
        assert evidence["static_support_intervals"] and not evidence["protected_motion_intervals"]
        assert set(row["metrics"]["source_pair"]["field_intervals"]) == {
            "action.cartesian_position",
            "action.gripper_position",
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
        }
    assert all(row["decision"] == "keep" for row in tables["episode_decisions"])


def test_scan_does_not_depend_on_workers(droid_like, droid_run, tmp_path):
    config = ScanConfig.preset("droid")
    scan_dataset(droid_like, output=tmp_path, config=config, name="droid_like", workers=2)
    for table in ("episode_metrics", "interval_findings", "episode_decisions", "video_findings"):
        assert (tmp_path / "scan" / f"{table}.jsonl").read_bytes() == (
            droid_run / "scan" / f"{table}.jsonl"
        ).read_bytes()


def test_compile_without_policy_changes_nothing(droid_run, tmp_path):
    package = compile_package(droid_run, output=tmp_path / "facts")
    assert package.release_ready, package.release_blockers
    assert not package.records("frame_decisions") and package.manifest["coverage"]["loss_fraction"] == 0
    assert {row["semantic_task"] for row in package.records("group_distribution")} == {
        "task 0",
        "task 1",
        "task 2",
    }


@requires_ffmpeg
def test_cleaning_policy_excludes_static_frames_only(droid_package, synthetic):
    assert droid_package.release_ready, droid_package.release_blockers
    excluded: dict[int, list] = {}
    for row in droid_package.records("frame_decisions"):
        excluded.setdefault(row["episode_index"], []).append((row["start_frame"], row["end_frame"]))
    assert set(excluded) == {0, 2}
    for episode, spans in excluded.items():
        _, static, joint_motion = synthetic.DROID_EPISODES[episode]
        frames = _frames(spans)
        assert frames <= set(range(*static))
        moving_joints = set(range(joint_motion[0] - 1, joint_motion[1])) if joint_motion else set()
        assert not frames & moving_joints, "joint-action motion vetoes the exclusion"
        assert len(frames) >= 0.9 * (static[1] - static[0] - len(moving_joints & set(range(*static))))
    policy = droid_package.records("policy_decisions")
    assert any(
        "auxiliary_action_motion" in row["reasons"] for row in policy if row["status"] == "vetoed_keep"
    )


def test_human_decisions(droid_run, tmp_path):
    _, tables = load_scan(droid_run)
    pause = next(row for row in tables["interval_findings"] if row["rule_id"] == "TRJ.NEAR_STATIC_RUN")
    start, end = pause["start_frame"] + 2, pause["start_frame"] + 10
    decisions = [
        {
            "candidate_id": pause["finding_id"],
            "action": "mask_field_interval",
            "field": "action.cartesian_position",
            "start_frame": start,
            "end_frame": end,
            "reason": "reviewed",
        }
    ]
    package = compile_package(droid_run, output=tmp_path / "human", decisions=decisions)
    assert package.records("field_masks") == [
        {
            "episode_index": pause["episode_index"],
            "start_frame": start,
            "end_frame": end,
            "field": "action.cartesian_position",
            "decision": "mask_field",
            "rules": ["TRJ.NEAR_STATIC_RUN"],
            "origin": "human",
            "candidate_id": pause["finding_id"],
            "reason": "reviewed",
        }
    ]
    with pytest.raises(ValueError, match="cannot express"):
        materialize(package, source_root=droid_run, output_root=tmp_path / "out")

    def compile_with(**changes):
        return compile_package(droid_run, output=tmp_path / "bad", decisions=[{**decisions[0], **changes}])

    with pytest.raises(ValueError, match="unknown candidate_id"):
        compile_with(candidate_id="0" * 24)
    with pytest.raises(ValueError, match="no evidence"):
        compile_with(field="action.joint_position")
    with pytest.raises(ValueError, match="inside the finding"):
        compile_with(end_frame=pause["end_frame"] + 1)
    with pytest.raises(ValueError, match="reason"):
        compile_with(reason="")
    quarantine = compile_package(
        droid_run,
        output=tmp_path / "quarantine",
        decisions=[
            {"candidate_id": pause["finding_id"], "action": "quarantine_episode", "reason": "reviewed"}
        ],
        max_coverage_loss=0.2,
    )
    assert not quarantine.release_ready and quarantine.release_blockers[0].startswith("coverage_loss:")
    rows = {row["episode_index"]: row for row in quarantine.records("episode_decisions")}
    assert rows[pause["episode_index"]]["decision"] == "quarantine"


@requires_ffmpeg
def test_materialized_dataset_matches_its_source(droid_like, droid_package, tmp_path):
    output = tmp_path / "clean"
    summary = materialize(
        droid_package, source_root=droid_like, output_root=output, rotation_stats_source="droid"
    )
    segments = plan_segments(droid_package)
    assert summary["output_episodes"] == len(segments) == 5
    assert summary["output_frames"] == sum(segment.length for segment in segments)
    for camera in DROID_CAMERAS:
        assert (output / "videos" / camera / "chunk-000" / "file-000.mp4").is_symlink()
    assert (output / "meta" / "rotation_stats.json").is_file()
    assert (
        json.loads((output / "meta" / "cleaning" / "quality_package.json").read_text())
        == droid_package.manifest
    )

    cleaned = LeRobotDataset("local/clean", root=output)
    source = LeRobotDataset("local/droid_like", root=droid_like)
    assert cleaned.num_episodes == len(segments) and cleaned.meta.total_frames == summary["output_frames"]
    starts, ends = validate_dense_layout(cleaned, output, np.arange(cleaned.num_episodes))
    assert ends[-1] == cleaned.num_frames
    stats = json.loads((output / "meta" / "stats.json").read_text())
    assert stats["action.cartesian_position"]["count"] == [cleaned.num_frames]
    for segment in segments:
        source_start = (
            int(source.meta.episodes[segment.source_episode]["dataset_from_index"]) + segment.source_start
        )
        for offset in (0, segment.length // 2, segment.length - 1):
            ours, theirs = cleaned[segment.from_index + offset], source[source_start + offset]
            assert int(ours["episode_index"]) == segment.episode and int(ours["frame_index"]) == offset
            assert torch.equal(ours["action.cartesian_position"], theirs["action.cartesian_position"])
            for camera in DROID_CAMERAS:
                assert torch.allclose(ours[camera], theirs[camera], atol=1e-6)

    with pytest.raises(FileExistsError):
        materialize(droid_package, source_root=droid_like, output_root=output)
    with pytest.raises(ValueError, match="inside the source"):
        materialize(droid_package, source_root=droid_like, output_root=droid_like / "clean")


def test_materialize_refuses_packages_with_blockers(droid_like, droid_run, tmp_path):
    package = compile_package(droid_run, output=tmp_path / "blocked", max_coverage_loss=-1.0)
    assert not package.release_ready
    with pytest.raises(ValueError, match="not release-ready"):
        materialize(package, source_root=droid_like, output_root=tmp_path / "out")
    tables = {name: package.records(name) for name in PACKAGE_TABLES}
    tables["episode_metrics"][0] = {**tables["episode_metrics"][0], "frame_count": 1}
    other = QualityPackage.write(tmp_path / "other", dataset=package.dataset, tables=tables)
    with pytest.raises(ValueError, match="another dataset"):
        materialize(other, source_root=droid_like, output_root=tmp_path / "out")


@requires_ffmpeg
def test_training_binding_of_a_scanned_package(droid_like, droid_package):
    meta = LeRobotDatasetMetadata("local/droid_like", root=droid_like)
    index = load_quality_index(
        droid_package.root,
        meta=meta,
        action_fields=["action.cartesian_position", "action.gripper_position"],
        observation_fields=[
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
            *DROID_CAMERAS,
        ],
        action_offsets=range(10),
        observation_offsets=[0],
        config=QualityConfig(),
    )
    episodes = meta.episodes.with_format("numpy")
    starts, ends = episodes["dataset_from_index"], episodes["dataset_to_index"]
    valid = _frames(index.valid_intervals(episodes["episode_index"], starts, ends, drop_n_last_frames=9))
    excluded = {
        int(starts[row["episode_index"]]) + frame
        for row in droid_package.records("frame_decisions")
        for frame in range(row["start_frame"], row["end_frame"])
    }
    windows = _frames((int(start), int(end) - 9) for start, end in zip(starts, ends, strict=True))
    # A frame keeps its own target, so every frame that is not excluded remains a valid sample.
    assert excluded and valid == windows - excluded


def test_agibot_useful_intervals_and_action_config(agibot_like, tmp_path):
    config = dataclasses.replace(
        ScanConfig.preset("agibot"),
        video=VideoConfig(enabled=False),
        visual_motion=VisualMotionConfig(enabled=False),
    )
    scan_dataset(agibot_like, output=tmp_path, config=config)
    package = compile_package(
        tmp_path, cleaning_policy=PausePolicyConfig(require_visual_evidence=False), max_coverage_loss=0.0
    )
    assert package.release_ready, package.release_blockers
    excluded = {}
    for row in package.records("frame_decisions"):
        excluded.setdefault(row["episode_index"], []).append((row["start_frame"], row["end_frame"]))
    assert excluded == {0: [(0, 10), (39, 150), (190, 200)], 2: [(19, 140)]}

    output = tmp_path / "clean"
    materialize(package, source_root=agibot_like, output_root=output)
    phases = {}
    for path in sorted((output / "meta" / "episodes").glob("*/*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "length", "action_config"]).to_pylist():
            phases[row["episode_index"]] = (
                row["length"],
                [(p["start_frame"], p["end_frame"]) for p in row["action_config"]],
            )
    assert phases == {
        0: (29, [(0, 29)]),
        1: (40, [(0, 40)]),
        2: (120, []),
        3: (19, [(0, 19)]),
        4: (10, [(0, 10)]),
    }


def test_scan_config_overrides(tmp_path):
    path = tmp_path / "scan.yaml"
    path.write_text(
        "contract: calvin\ntrajectory:\n  pause_duration_seconds: [2.0]\nvideo:\n  enabled: false\n"
    )
    config = load_scan_config(path)
    assert config.trajectory.pause_duration_seconds == (2.0,) and config.trajectory.check_jerk
    assert not config.video.enabled and config.alignment.check_absolute_next
    with pytest.raises(ValueError, match="disagrees"):
        load_scan_config(path, contract="droid")
    path.write_text("contract: droid\ntrajectory:\n  pause_seconds: [2.0]\n")
    with pytest.raises(ValueError, match="Unknown scan.trajectory options"):
        load_scan_config(path)
    with pytest.raises(ValueError, match="Unknown source contract"):
        load_scan_config(None, contract="bridge")


def test_video_probe_findings(droid_like, tmp_path):
    item = video.VideoFile(
        "videos/observation.images.wrist_left/chunk-000/file-000.mp4", "observation.images.wrist_left"
    )
    config = VideoConfig(sample_count_per_file=4)
    expected = {"height": 48, "width": 64, "fps": 15.0}
    probe = video.probe_and_judge(droid_like, item, config, expected)
    assert probe["decode_ok"] and probe["findings"] == []
    wrong = video.probe_findings(probe, expected={**expected, "fps": 30.0, "width": 32}, config=config)
    assert {row["rule_id"] for row in wrong} == {"VID.FRAME_RATE_MISMATCH", "VID.FRAME_SHAPE_MISMATCH"}
    broken = tmp_path / "videos" / "cam" / "file.mp4"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a video")
    probe = video.probe_and_judge(tmp_path, video.VideoFile("videos/cam/file.mp4", "cam"), config, None)
    assert [row["rule_id"] for row in probe["findings"]] == ["VID.DECODE_FAILURE"]


@requires_ffmpeg
def test_cli(droid_like, tmp_path, capsys):
    cli.main(["scan", "--dataset", str(droid_like), "--output", str(tmp_path), "--contract", "droid"])
    cli.main(["compile", "--run", str(tmp_path), "--cleaning-policy"])
    assert '"release_ready": true' in capsys.readouterr().out
    cli.main(
        [
            "materialize",
            "--package",
            str(tmp_path / "package"),
            "--dataset",
            str(droid_like),
            "--output",
            str(tmp_path / "clean"),
            "--media",
            "copy",
        ]
    )
    out = capsys.readouterr().out
    assert '"output_episodes": 5' in out
    assert not (tmp_path / "clean" / "videos" / DROID_CAMERAS[0] / "chunk-000" / "file-000.mp4").is_symlink()


def test_missing_ffmpeg_is_reported(droid_like, tmp_path, caplog):
    preset = ScanConfig.preset("droid")
    missing = dataclasses.replace(preset.visual_motion, ffmpeg_binary="concertina-test-no-such-ffmpeg")
    config = dataclasses.replace(preset, video=VideoConfig(enabled=False), visual_motion=missing)
    with caplog.at_level(logging.WARNING, logger="concertina.data_quality.pipeline"):
        summary = scan_dataset(droid_like, output=tmp_path, config=config)
    assert len(summary["warnings"]) == 1 and "keeps every pause" in summary["warnings"][0]
    assert summary["warnings"][0] in caplog.text
    _, tables = load_scan(tmp_path)
    pauses = [row for row in tables["interval_findings"] if row["rule_id"] == "TRJ.NEAR_STATIC_RUN"]
    assert pauses
    for row in pauses:
        statuses = {camera["status"] for camera in row["visual_motion_evidence"]["camera_evidence"]}
        assert statuses == {"ffmpeg_unavailable"}
    package = compile_package(tmp_path, cleaning_policy=PausePolicyConfig())
    assert package.manifest["scan"]["warnings"] == summary["warnings"]
    assert not package.records("frame_decisions"), "without camera evidence every pause is kept"
    strict = dataclasses.replace(config, visual_motion=dataclasses.replace(missing, require_ffmpeg=True))
    with pytest.raises(RuntimeError, match="not on PATH"):
        scan_dataset(droid_like, output=tmp_path / "strict", config=strict)
