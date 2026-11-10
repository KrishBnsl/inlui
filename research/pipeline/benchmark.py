"""Blocked legacy inference benchmark.

The former script mixed the availability-invalid model, heuristic admission
probabilities, and hard-coded scalability statements. It is retained as a
fail-fast compatibility entry point so old commands cannot create new evidence.
Use :mod:`research.pipeline.research_evaluation` for executable, hash-bound metrics.
"""


BLOCK_MESSAGE = (
    "BLOCKED: research/pipeline/benchmark.py used the invalidated legacy methodology and "
    "contained unmeasured performance claims. Run `.venv/bin/python "
    "-m research.pipeline.research_evaluation --help` and use the command in "
    "docs/TECHNICAL_REPORT.md#reproducibility instead."
)


def main() -> None:
    raise SystemExit(BLOCK_MESSAGE)


if __name__ == "__main__":
    main()
