"""Tests for run store registry and retention."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from options_surface_edge_engine.storage.run_store import (
    build_run_metadata,
    delete_run,
    list_runs,
    next_run_label,
    prune_runs,
    register_run,
)


def test_register_run_increments_label(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    label1 = next_run_label(output_dir, prefix="Output")
    assert label1 == "Output-0001"
    meta1 = build_run_metadata("r1", label1, ["SPY"], "delayed_eod", best_trade=None)
    register_run(output_dir, meta1)
    label2 = next_run_label(output_dir, prefix="Output")
    assert label2 == "Output-0002"


def test_delete_run_removes_registry_and_folder(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    meta = build_run_metadata("r2", "Output-0001", ["SPY"], "delayed_eod", best_trade=None)
    register_run(output_dir, meta)
    run_dir = output_dir / "runs" / "r2"
    assert run_dir.exists()
    ok = delete_run(output_dir, "r2")
    assert ok is True
    assert not run_dir.exists()
    runs = list_runs(output_dir)
    assert runs.empty


def test_prune_keeps_latest_n(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    for idx in range(1, 6):
        label = f"Output-{idx:04d}"
        meta = build_run_metadata(f"r{idx}", label, ["SPY"], "delayed_eod", best_trade=None)
        register_run(output_dir, meta)
    removed = prune_runs(output_dir, keep_last=2)
    assert len(removed) == 3
    runs = list_runs(output_dir)
    assert len(runs) == 2
