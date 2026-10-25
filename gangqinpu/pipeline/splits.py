"""Group-disjoint train/validation/test assignment.

Whole split groups are allocated (largest first, seeded tie-breaking) so that
per-source sample counts approach their targets while the weak genre labels of
the primary corpus stay approximately stratified; whole-group moves then
repair source counts. Groups can be pinned to the test split (e.g. groups
containing the test pieces of an external benchmark).
"""
from __future__ import annotations

import numpy as np

SPLITS = ["train", "val", "test"]
RATIOS = np.array([0.8, 0.1, 0.1])
GENRES = ["classical", "pop", "jazz", "folk_traditional", "blues", "rock", "other", "unknown"]


def _targets(n: int) -> np.ndarray:
    raw = RATIOS * n
    out = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - out), kind="stable")[:n - out.sum()]:
        out[i] += 1
    return out


def assign_groups(
    groups: dict[str, dict],
    seed: int = 0,
    heldout_size: int | None = None,
    protected_test: set[str] | frozenset = frozenset(),
) -> tuple[dict[str, str], dict]:
    """Assign each group to a split.

    ``groups`` maps group id -> {"primary": n, "reference": n, "genres": Counter}.
    ``heldout_size`` fixes the primary corpus's validation and test sizes;
    otherwise 80/10/10 ratios are used. Reference-corpus counts follow 80/10/10.
    """
    rng = np.random.default_rng(seed)
    vectors = {
        name: np.array([g["primary"], g["reference"], *[g["genres"].get(label, 0) for label in GENRES]], dtype=int)
        for name, g in groups.items()
    }
    population = sum(vectors.values(), np.zeros(2 + len(GENRES), dtype=int))
    goal = RATIOS[:, None] * population[None, :]
    goal[:, :2] = np.stack([_targets(int(n)) for n in population[:2]], axis=1)
    if heldout_size is not None:
        n = int(population[0])
        if not 0 < 2 * heldout_size < n:
            raise ValueError("heldout_size must leave samples for training")
        goal[:, 0] = [n - 2 * heldout_size, heldout_size, heldout_size]
        goal[:, 2:] = (goal[:, 0] / n)[:, None] * population[None, 2:]

    counts = np.zeros_like(goal, dtype=int)
    assignment: dict[str, str] = {}
    for name in sorted(protected_test):
        assignment[name] = "test"
        counts[2] += vectors[name]

    # Normalise each source's objective so the larger corpus does not dominate.
    weights = np.r_[4 / np.maximum(population[:2], 1), np.ones(len(GENRES)) / max(population[0], 1)]

    def delta(i: int, v: np.ndarray) -> float:
        return float(np.sum(weights * ((counts[i] + v - goal[i]) ** 2 - (counts[i] - goal[i]) ** 2)
                            / np.maximum(goal[i], 1)))

    jitter = {name: float(rng.random()) for name in sorted(vectors)}
    free = [name for name in vectors if name not in assignment]
    order = sorted(free, key=lambda k: (-max(vectors[k][:2] / np.maximum(population[:2], 1)), jitter[k], k))
    for name in order:
        v = vectors[name]
        allowed = [i for i in range(3) if np.all(counts[i, :2] + v[:2] <= goal[i, :2])]
        i = min(allowed or range(3), key=lambda i: (delta(i, v), i))
        assignment[name] = SPLITS[i]
        counts[i] += v

    # Whole-group moves only, and only if they reduce the source count error.
    for _ in range(100):
        moves = []
        for name in order:
            v, i = vectors[name], SPLITS.index(assignment[name])
            for j in range(3):
                if i == j:
                    continue
                before = np.abs(counts[[i, j], :2] - goal[[i, j], :2]).sum()
                after = (np.abs(counts[i, :2] - v[:2] - goal[i, :2]).sum()
                         + np.abs(counts[j, :2] + v[:2] - goal[j, :2]).sum())
                if after < before:
                    moves.append((after - before, delta(i, -v) + delta(j, v), name, i, j))
        if not moves:
            break
        _, _, name, i, j = min(moves)
        counts[i] -= vectors[name]
        counts[j] += vectors[name]
        assignment[name] = SPLITS[j]

    report = {
        "targets": {src: dict(zip(SPLITS, map(int, goal[:, k]))) for k, src in enumerate(["primary", "reference"])},
        "actual": {src: dict(zip(SPLITS, map(int, counts[:, k]))) for k, src in enumerate(["primary", "reference"])},
        "genres": {split: dict(zip(GENRES, map(int, counts[i, 2:]))) for i, split in enumerate(SPLITS)},
    }
    return assignment, report
