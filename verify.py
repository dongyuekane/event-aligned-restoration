"""Check archived result summaries and optionally compare a generated replay."""

import argparse
import csv
import json
from math import isclose
from pathlib import Path
from statistics import median

from study import ARCHIVE, METHODS, summarize


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_policy_results(directory, split):
    result = rows(directory / f"{split}_results.csv")
    for item in result:
        for key in ("ens_kwh", "policy_evaluation_s"):
            item[key] = float(item[key])
    return result


def verify(replay):
    protocol = json.loads((ARCHIVE / "protocol.json").read_text())
    for split in ("development", "confirmation"):
        policy = load_policy_results(ARCHIVE, split)
        saved = json.loads((ARCHIVE / f"{split}_metrics.json").read_text())
        if len(policy) != len(METHODS) * protocol[f"n_{split}"]:
            raise ValueError(f"Unexpected number of archived {split} policy rows.")
        actual = summarize(policy)
        for method in METHODS:
            for key, value in actual[method].items():
                if not isclose(value, saved[method][key], abs_tol=1e-9):
                    raise ValueError(f"Archived metric mismatch in {split}: {method} {key}")

        if replay:
            regenerated = load_policy_results(replay, split)
            if len(regenerated) != len(policy):
                raise ValueError(f"Incorrect number of replayed {split} rows.")
            for original, rerun in zip(policy, regenerated):
                fields = set(original) - {"policy_evaluation_s", "ens_kwh"}
                if (any(original[key] != rerun[key] for key in fields)
                        or not isclose(original["ens_kwh"], rerun["ens_kwh"], rel_tol=0, abs_tol=1e-10)):
                    raise ValueError(f"Policy replay differs in {split}: {original['case']} {original['method']}")
        print(f"{split}: verified {len(policy)} full-policy results")

    timing = rows(ARCHIVE / "timing.csv")
    timing_summary = json.loads((ARCHIVE / "timing_metrics.json").read_text())
    for method in METHODS:
        elapsed = [int(item["elapsed_ns"]) / 1e6 for item in timing if item["method"] == method]
        if len(elapsed) != 3 * protocol["n_confirmation"] or not isclose(median(elapsed), timing_summary[method]["median_ms"]):
            raise ValueError(f"Archived timing mismatch for {method}.")
    if replay:
        replay_timing = rows(replay / "timing.csv")
        decisions = ("case", "rep", "method", "action")
        if [tuple(r[k] for k in decisions) for r in timing] != [tuple(r[k] for k in decisions) for r in replay_timing]:
            raise ValueError("Root decisions differ in the replayed timing run.")
    print(f"timing: verified {len(timing)} root decisions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path)
    verify(parser.parse_args().replay)
