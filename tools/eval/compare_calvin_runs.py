"""Compare the per-sequence outcomes of CALVIN runs.

Accepts `summary.json` files with per-sequence rows (`sequences` from `concertina-eval`, or `sequence_results` /
`summary.sequence_results` lists), and `results.jsonl` files or directories holding them. Prints, for every pair
of runs, the number of common sequences, how many of them have the same successful length, and both average
lengths over the common sequences.

    python tools/eval/compare_calvin_runs.py a=<run a>/summary.json b=<run b>/summary.json
"""

import argparse
import itertools
import json
from pathlib import Path


def load_outcomes(path: Path) -> dict[int, int]:
    if path.is_dir():
        files = sorted(path.rglob("results.jsonl"))
        if not files:
            raise FileNotFoundError(f"No results.jsonl under {path}")
        return {index: length for file in files for index, length in load_outcomes(file).items()}
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text())
        rows = (
            payload.get("sequences")
            or payload.get("sequence_results")
            or payload.get("summary", {}).get("sequence_results")
        )
        if rows is None:
            raise ValueError(f"{path} holds no per-sequence results")
    return {int(row["sequence_index"]): int(row["success_len"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("runs", nargs="+", help="name=path entries")
    args = parser.parse_args()
    runs = {}
    for entry in args.runs:
        name, _, path = entry.partition("=")
        runs[name] = load_outcomes(Path(path))
    for (name_a, a), (name_b, b) in itertools.combinations(runs.items(), 2):
        common = sorted(set(a) & set(b))
        if not common:
            print(f"{name_a} vs {name_b}: no common sequences")
            continue
        same = sum(a[i] == b[i] for i in common)
        mean_a = sum(a[i] for i in common) / len(common)
        mean_b = sum(b[i] for i in common) / len(common)
        print(
            f"{name_a} vs {name_b}: {len(common)} common sequences, {same} identical outcomes "
            f"({same / len(common):.1%}), average length {mean_a:.3f} vs {mean_b:.3f}"
        )


if __name__ == "__main__":
    main()
