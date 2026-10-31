"""Replay frozen report actions on a label-independent internal partition.

This script never fits a model or changes a threshold. It writes aggregate
counts only; no IDs or group identifiers are written to the output files.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "uteg_nc_goal_20261030"
OLD = OUT / "../uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551"
SEQ = OUT / "../uteg_current_ma_sequence_20261026/actions"
EXT = OUT / "../uteg_ridge_dual_head_external_eval_20261026_191117/actions"
DATA = OUT / "../uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz"
INC = OUT / "../uteg_reporting_increment_20261030"
DIRECTIONS = ["HZ_to_SA", "HZ+HB_to_SA"]
SEEDS = np.arange(1000, 1020, dtype=np.int32)
SCHEDULES = {"5-10-15": [300, 600, 900], "10-15": [600, 900], "15": [900]}
PARTITION = json.loads((INC / "subset_mask.json").read_text(encoding="utf-8"))
SELECTED = np.asarray(PARTITION["selected_global_indices"], dtype=int)
COMPLEMENT = np.asarray(PARTITION["complement_global_indices"], dtype=int)

HASHES = {}


def remember(path: Path) -> Path:
    path = path.resolve()
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def load_npz(path: Path):
    with np.load(remember(path), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def table(name, rows):
    assert rows
    with (HERE / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, groups, draws=5000, seed=20261030):
    """Group-bootstrap interval for an event proportion."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    num = np.asarray(values, float)
    den = np.ones_like(num, float)
    group_num = np.asarray([num[groups == g].sum() for g in unique])
    group_den = np.asarray([den[groups == g].sum() for g in unique])
    sample = np.random.default_rng(seed).integers(0, len(unique), size=(draws, len(unique)))
    bnum = group_num[sample].sum(axis=1)
    bden = group_den[sample].sum(axis=1)
    point = float(num.sum() / den.sum())
    q = np.quantile(bnum / bden, [0.025, 0.975])
    return {"estimate": point, "lo": float(q[0]), "hi": float(q[1]), "valid_resamples": draws}


def state_from_med(med):
    return np.where(med < 50, -1, np.where(med > 70, 1, 0)).astype(np.int8)


def score_partition(direction, schedule_name, checkpoints, truth, groups, mask, partition_name):
    """Run the frozen first-report state machine for one partition."""
    schedule = SCHEDULES[schedule_name]
    released = np.zeros((20, len(truth)), dtype=bool)
    first_class = np.full((20, len(truth)), 9, dtype=np.int8)
    first_time = np.full((20, len(truth)), -1, dtype=np.int16)
    for horizon in schedule:
        actions, predictions = checkpoints[horizon]
        new = actions & ~released
        first_class[new] = predictions[new]
        first_time[new] = horizon
        released |= actions
    # First-report locking is an explicit state invariant.
    assert np.all((first_time >= 0) == released)
    assert np.all((first_class == 9) == ~released)
    correct = released & (first_class == truth)
    incorrect = released & (first_class != truth)
    unreported = ~released
    n_records = int(mask.sum())
    n_events = n_records * len(SEEDS)
    assert int(correct.sum() + incorrect.sum() + unreported.sum()) == n_events

    group_events = np.repeat(groups, len(SEEDS)).reshape(len(groups), len(SEEDS)).T.reshape(-1)
    # The arrays flatten seed-major; repeat each group for 20 seeds, then tile
    # records for each seed to match the seed-major order.
    group_events = np.tile(groups, len(SEEDS))
    rows = []
    for name, values in [("correct", correct), ("incorrect", incorrect), ("unreported", unreported), ("released", released)]:
        values = values.reshape(-1)
        boot = interval(values, group_events)
        rows.append({
            "direction": direction,
            "schedule": schedule_name,
            "partition": partition_name,
            "records": n_records,
            "events": n_events,
            "outcome": name,
            "count": int(values.sum()),
            "fraction": float(values.mean()),
            "ci_lo": boot["lo"],
            "ci_hi": boot["hi"],
        })
    seed_rows = []
    for k, seed in enumerate(SEEDS):
        released_n = int(released[k].sum())
        seed_rows.append({
            "direction": direction,
            "schedule": schedule_name,
            "partition": partition_name,
            "seed": int(seed),
            "correct": int(correct[k].sum()),
            "incorrect": int(incorrect[k].sum()),
            "unreported": int(unreported[k].sum()),
            "released": released_n,
            "coverage": released_n / n_records,
            "agreement": int(correct[k].sum()) / released_n if released_n else None,
        })
    destination_rows = []
    for true_state in [-1, 0, 1]:
        denominator = int((truth == true_state).sum() * len(SEEDS))
        class_mask = np.broadcast_to(truth == true_state, first_class.shape)
        for report_state in [-1, 0, 1, 9]:
            count = int((class_mask & (first_class == report_state)).sum())
            destination_rows.append({
                "direction": direction,
                "schedule": schedule_name,
                "partition": partition_name,
                "true_state": true_state,
                "report_state": report_state,
                "count": count,
                "denominator": denominator,
                "fraction": count / denominator if denominator else None,
            })
    time_rows = []
    for horizon in [300, 600, 900, -1]:
        count = int((first_time == horizon).sum())
        time_rows.append({
            "direction": direction,
            "schedule": schedule_name,
            "partition": partition_name,
            "time_seconds": horizon,
            "count": count,
            "denominator": n_events,
        })
    zero_report = int((released.sum(axis=1) == 0).sum())
    aggregate = {
        "direction": direction,
        "schedule": schedule_name,
        "partition": partition_name,
        "records": n_records,
        "events": n_events,
        "correct": int(correct.sum()),
        "incorrect": int(incorrect.sum()),
        "unreported": int(unreported.sum()),
        "coverage": float(released.mean()),
        "agreement": float(correct.sum() / released.sum()) if released.sum() else None,
        "zero_report_seeds": zero_report,
    }
    return aggregate, rows, seed_rows, destination_rows, time_rows, (released, first_class, first_time)


def main():
    # The output files are deterministic aggregates for this scoped replay.
    remember(HERE / "ANALYSIS_PLAN.md")
    remember(INC / "subset_mask.json")
    data = load_npz(DATA)
    assert len(SELECTED) == 60 and len(COMPLEMENT) == 141
    assert not set(SELECTED) & set(COMPLEMENT)
    groups_all = data["groups"]
    selected_groups_global = set(groups_all[SELECTED].tolist())
    complement_groups_global = set(groups_all[COMPLEMENT].tolist())
    assert not selected_groups_global & complement_groups_global
    assert len(selected_groups_global) == 57 and len(complement_groups_global) == 133
    partitions = {"full201": None, "selected60": SELECTED, "complement141": COMPLEMENT}
    results = []
    metric_rows = []
    seed_rows = []
    destination_rows = []
    time_rows = []
    checks = []
    qc_rows = []
    for direction in DIRECTIONS:
        t300 = load_npz(SEQ / "T300" / f"{direction}.npz")
        t600 = load_npz(SEQ / "T600" / f"{direction}.npz")
        t900 = load_npz(EXT / f"{direction}.npz")
        assert np.array_equal(t300["seed"], SEEDS) and np.array_equal(t600["seed"], SEEDS)
        assert np.array_equal(t300["ids"], t600["ids"]) and np.array_equal(t300["ids"], t900["test_ids"])
        assert np.array_equal(t300["groups"], t600["groups"])
        idx = t300["index"]
        assert len(idx) == 201 and np.array_equal(np.sort(idx), np.sort(np.unique(idx)))
        truth = state_from_med(data["medical"][idx, 2])
        assert np.array_equal(t300["groups"], groups_all[idx])
        checkpoints = {300: (t300["action"].astype(bool), t300["pred"]), 600: (t600["action"].astype(bool), t600["pred"]), 900: (t900["unified_action"].astype(bool), t900["final_ma_class"])}
        qc_total = int(data["input_qc"][idx].size)
        qc_true = int(data["input_qc"][idx].sum())
        qc_rows.extend([
            {"direction": direction, "checkpoint": 300, "input_qc_true": qc_true, "input_qc_total": qc_total, "action_valid_true": int(t300["valid"].sum()), "action_valid_total": int(t300["valid"].size)},
            {"direction": direction, "checkpoint": 600, "input_qc_true": qc_true, "input_qc_total": qc_total, "action_valid_true": int(t600["valid"].sum()), "action_valid_total": int(t600["valid"].size)},
            {"direction": direction, "checkpoint": 900, "input_qc_true": int(t900["input_qc"].sum()), "input_qc_total": int(t900["input_qc"].size), "action_valid_true": int(t900["dual_valid"].sum()), "action_valid_total": int(t900["dual_valid"].size)},
        ])
        original = load_npz(SEQ / f"{direction}_first_report.npz")
        full = score_partition(direction, "5-10-15", checkpoints, truth, groups_all[idx], np.ones(201, dtype=bool), "full201")
        assert np.array_equal(full[-1][0], original["reported"])
        assert np.array_equal(full[-1][1], original["first_class"])
        assert np.array_equal(full[-1][2], original["first_time"])
        for schedule_name in SCHEDULES:
            for partition_name, indices in partitions.items():
                if indices is None:
                    indices = np.arange(201)
                else:
                    # The manifest stores DATA15 global indices, whereas the
                    # frozen action arrays are ordered in local test-cohort order.
                    indices = np.asarray([int(np.flatnonzero(idx == g)[0]) for g in indices], dtype=int)
                mask = np.ones(len(indices), dtype=bool)
                sub_truth = truth[indices]
                sub_groups = groups_all[idx][indices]
                sub_checkpoints = {h: (a[:, indices], p[:, indices]) for h, (a, p) in checkpoints.items()}
                result = score_partition(direction, schedule_name, sub_checkpoints, sub_truth, sub_groups, mask, partition_name)
                aggregate, metrics, seeds, destinations, times, state = result
                results.append(aggregate)
                metric_rows.extend(metrics)
                seed_rows.extend(seeds)
                destination_rows.extend(destinations)
                time_rows.extend(times)
                if partition_name == "full201" and schedule_name == "5-10-15":
                    assert np.array_equal(state[0], original["reported"])
                checks.append({"direction": direction, "schedule": schedule_name, "partition": partition_name, "flow_closed": aggregate["correct"] + aggregate["incorrect"] + aggregate["unreported"] == aggregate["events"], "first_reports_locked": True})
    # Full event accounting must equal the two disjoint partitions.
    for direction in DIRECTIONS:
        for schedule_name in SCHEDULES:
            full = next(x for x in results if x["direction"] == direction and x["schedule"] == schedule_name and x["partition"] == "full201")
            selected = next(x for x in results if x["direction"] == direction and x["schedule"] == schedule_name and x["partition"] == "selected60")
            complement = next(x for x in results if x["direction"] == direction and x["schedule"] == schedule_name and x["partition"] == "complement141")
            for key in ["correct", "incorrect", "unreported"]:
                assert full[key] == selected[key] + complement[key]
    before = dict(HASHES)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in before.items())
    dump("RESULTS.json", {"validation_level": "internal_partition_robustness_check", "results": results})
    table("event_metrics.csv", metric_rows)
    table("seed_metrics.csv", seed_rows)
    table("class_destinations.csv", destination_rows)
    table("first_report_times.csv", time_rows)
    table("qc_summary.csv", qc_rows)
    dump("INPUT_HASHES.json", HASHES)
    dump("CHECKS.json", {"status": "PASS", "new_fits": 0, "new_thresholds": 0, "partition_group_disjoint": True, "partition_record_exhaustive": True, "full_replay_matches_saved_first_report": True, "flow_closed": True, "first_reports_locked": True, "input_hashes_unchanged": True, "resamples": 5000, "records": {"full201": 201, "selected60": 60, "complement141": 141}, "proxy_groups": {"full201": 190, "selected60": 57, "complement141": 133}})
    print(json.dumps({"status": "PASS", "results": len(results), "metric_rows": len(metric_rows), "seed_rows": len(seed_rows)}, indent=2))


if __name__ == "__main__":
    main()
