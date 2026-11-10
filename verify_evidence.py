"""Recompute FlipTrace-GC's released statistics and replay its access traces."""

from collections import OrderedDict
import json
from math import isclose
from pathlib import Path

from fliptrace_gc.policy import lru_prefix_reuse, schedule_prefix_reuse


EVIDENCE = Path(__file__).resolve().parent / "evidence"
UPDATES = 400


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(filename):
    data = json.loads((EVIDENCE / filename).read_text(encoding="utf-8"))
    require(data["schema_version"] == 1, f"{filename}: unsupported evidence schema")
    return data


def count_events(plan):
    return {
        "full_hit_batches": sum(full_hit for full_hit, _, _ in plan),
        "written_views": sum(len(admit) for _, _, admit in plan),
    }


def verify_traces(data, batch_size):
    slots = data["replay_slots"]
    require(slots == 45, "Unexpected replay capacity")
    traces = data["traces"]
    require(set(traces) == {"A-3407", "A-304", "B-3407", "official_train-3407"},
            "Unexpected trace set")
    replays = {}
    for name, trace in traces.items():
        require(name == f"{trace['subset']}-{trace['seed']}", f"{name}: inconsistent trace name")
        image_count = 367 if trace["subset"] == "official_train" else 37
        require(trace["training_images"] == image_count, f"{name}: unexpected training size")
        require(len(trace["batches"]) == UPDATES, f"{name}: unexpected number of updates")
        batches = []
        for update, batch in enumerate(trace["batches"], 1):
            require(len(batch) == batch_size, f"{name}: unexpected batch size at update {update}")
            for index, flip in batch:
                require(type(index) is int and 0 <= index < image_count and type(flip) is bool,
                        f"{name}: invalid view key at update {update}")
            batches.append(tuple(map(tuple, batch)))

        recency = OrderedDict()
        observed = {
            "schedule": count_events(schedule_prefix_reuse(batches, slots)),
            "lru": count_events([lru_prefix_reuse(recency, batch, slots) for batch in batches]),
        }
        require(observed == trace["expected_replay"], f"{name}: replay counts differ")
        replays[name] = observed
        print(f"{name}: schedule {observed['schedule']['full_hit_batches']}/"
              f"{observed['schedule']['written_views']}, LRU "
              f"{observed['lru']['full_hit_batches']}/{observed['lru']['written_views']} hits/writes")
    return replays


def verify_pairs(data, traces, replays):
    pairs = data["pairs"]
    expected_traces = {
        "R1": "A-3407", "R2": "A-304", "L1": "A-304", "L2": "B-3407",
        "F": "A-3407", "E1": "A-3407", "E2": "A-304", "E3": "official_train-3407",
    }
    require({name: pair["trace"] for name, pair in pairs.items()} == expected_traces,
            "Unexpected timing pair and trace associations")
    for name, pair in pairs.items():
        arms = pair["arms"]
        require(len(pair["arm_order"]) == 2 and set(pair["arm_order"]) == set(arms),
                f"{name}: inconsistent arm order")
        reference = arms[pair["reference_arm"]]
        candidate = arms[pair["candidate_arm"]]
        require({pair["reference_arm"], pair["candidate_arm"]} == set(arms),
                f"{name}: inconsistent arm roles")
        require(all(arm["completed_updates"] == UPDATES and arm["timed_state_reset_verified"]
                    for arm in arms.values()), f"{name}: arms differ from the matched-update protocol")
        require(sum(arm["train_seconds"] for arm in arms.values())
                <= pair["elapsed_seconds"] <= pair["budget_seconds"],
                f"{name}: training and pair-wall-clock intervals are inconsistent")

        saving = 1 - candidate["train_seconds"] / reference["train_seconds"]
        delta = candidate["miou_percent"] - reference["miou_percent"]
        peak_ratio = candidate["train_peak_allocated_bytes"] / reference["train_peak_allocated_bytes"]
        reported = pair["reported"]
        require(isclose(saving, reported["time_reduction_fraction_including_setup"], abs_tol=1e-12),
                f"{name}: time reduction differs")
        require(isclose(delta, reported["ours_minus_reference_pp"], abs_tol=1e-12),
                f"{name}: validation mIoU difference differs")
        require(isclose(1 - peak_ratio, reported["memory_reduction_fraction"], abs_tol=1e-12),
                f"{name}: peak allocation difference differs")

        gate = pair["gate"]
        if "max_abs_accuracy_difference_pp" in gate:
            accuracy_passed = abs(delta) <= gate["max_abs_accuracy_difference_pp"]
        else:
            accuracy_passed = delta >= -gate["max_accuracy_drop_pp"]
        payload_passed = all(arm["cache_host_bytes"] <= gate["max_cache_payload_bytes"]
                             for arm in arms.values()) if "max_cache_payload_bytes" in gate else True
        resource_passed = (saving >= gate["min_time_reduction_including_cache_setup"]
                           and peak_ratio <= gate["max_peak_ratio"] and payload_passed)
        require(accuracy_passed == reported["accuracy_passed"]
                and resource_passed == reported["resource_passed"]
                and (accuracy_passed and resource_passed) == reported["passed"],
                f"{name}: gate outcome differs")

        bounded = pair.get("observed_bounded_cache", {})
        require(set(bounded) == set(arms) & {"schedule", "lru"},
                f"{name}: missing or unexpected observed cache counters")
        for policy, counters in bounded.items():
            require(replays[pair["trace"]][policy] == counters,
                    f"{name}: {policy} replay differs from the measured cache counters")
        if "reported_cache_views" in pair:
            trace = traces[pair["trace"]]
            draws = [tuple(key) for batch in trace["batches"] for key in batch]
            accessed = len(set(draws))
            views = {
                "prebuilt": 2 * trace["training_images"],
                "accessed": accessed,
                "draws": len(draws),
                "repeated_draws": len(draws) - accessed,
            }
            require(views == pair["reported_cache_views"], f"{name}: eager view counts differ")
        print(f"{name}: {reference['train_seconds']:.3f} to {candidate['train_seconds']:.3f} s, "
              f"{saving:+.2%}, validation {delta:+.6f} pp, "
              f"gate {'pass' if reported['passed'] else 'fail'}")


def verify_holdout(data, pairs):
    classes = data["classes"]
    require(classes == len(data["class_order"]) == 11 and data["void_id"] == 11,
            "Unexpected test class definition")
    require(data["completed_updates"] == UPDATES and data["test_images"] == 233
            and data["valid_pixels"] == 38_719_017, "Unexpected held-out scope")
    names = {"a304": "L1", "b3407": "L2"}
    arms = data["arms"]
    require(set(arms) == {f"{pair}_{policy}" for pair in names for policy in ("lru", "schedule")},
            "Unexpected held-out arms")
    ground_truth = None
    scores = {}
    for name, record in arms.items():
        pair, policy = name.split("_")
        validation = pairs[names[pair]]["arms"][policy]["miou_percent"]
        require(record["validation_miou_percent"] == validation,
                f"{name}: validation score differs from the matched timing arm")
        matrix = record["confusion_matrix"]
        require(len(matrix) == classes and all(len(row) == classes for row in matrix),
                f"{name}: unexpected matrix shape")
        require(all(type(count) is int and count >= 0 for row in matrix for count in row),
                f"{name}: invalid confusion count")
        actual = list(map(sum, matrix))
        require(sum(actual) == data["valid_pixels"], f"{name}: pixel total differs")
        if ground_truth is None:
            ground_truth = actual
        require(actual == ground_truth, f"{name}: true-class marginals differ")
        predicted = list(map(sum, zip(*matrix)))
        diagonal = [row[i] for i, row in enumerate(matrix)]
        unions = [row + column - hit for row, column, hit in zip(actual, predicted, diagonal)]
        iou = [hit / union for hit, union in zip(diagonal, unions) if union]
        scores[name] = 100 * sum(iou) / len(iou)
        require(isclose(scores[name], record["test_miou_percent"], abs_tol=1e-12),
                f"{name}: held-out mIoU differs")
        print(f"{name}: test mIoU {scores[name]:.6f}%")
    for pair in names:
        delta = scores[f"{pair}_schedule"] - scores[f"{pair}_lru"]
        contrast = data["contrasts"][f"{pair}_schedule_minus_lru"]
        require(isclose(delta, contrast["delta_pp"], abs_tol=1e-12)
                and (abs(delta) <= contrast["gate_pp"]) == contrast["passed"],
                f"{pair}: held-out comparison differs")
        print(f"{pair}: schedule minus LRU {delta:+.6f} pp")


def main():
    timing = load("paired-timing.json")
    trace_data = load("schedule-traces.json")
    print("Policy replay at 45 slots. Replay-only combinations are listed in evidence/README.md.")
    replays = verify_traces(trace_data, timing["protocol"]["train_batch_size"])
    verify_pairs(timing, trace_data["traces"], replays)
    verify_holdout(load("heldout-confusions.json"), timing["pairs"])
    print("Verified eight timing pairs, four held-out confusion matrices and four access traces.")


if __name__ == "__main__":
    main()
