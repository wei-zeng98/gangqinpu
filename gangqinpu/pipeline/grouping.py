"""Score-similarity grouping for leakage-aware splitting.

Scores are compared symbolically from their score MIDI:

1. Normalisation: merge non-drum tracks, quarter-note grid at 1/96 resolution,
   drop velocity/tempo, deduplicate (pitch, onset, duration) triples. Exact and
   transposition-invariant hashes define exact-score identities and
   transposition families.
2. Retrieval: outer voices (highest / lowest pitch per onset) are turned into
   transposition-invariant 8-interval shingles, TF-IDF weighted; every family
   pair with cosine >= 0.04 becomes a candidate (no top-k cut-off).
3. Verification: monotonic interval anchors with one consistent pitch shift
   and locally consistent rhythm, followed by one-to-one polyphonic note
   matching inside the anchored range. Pairs are labelled
   ``highly_similar``, ``excerpt_or_structural_variant``,
   ``related_or_uncertain`` or not related. Pairs involving a reference corpus
   (e.g. ASAP) additionally use an outer-voice melody comparison.
4. Groups: ``score_group_id`` is a deterministic complete-link partition of
   transposition families over ``highly_similar`` edges (every pair of
   families in a group is directly highly similar). ``group_id`` (split
   group) is the connected component over *all* related/uncertain edges and
   shared reference work IDs - intentionally conservative.

These are operational groups, not verified composition identities.
"""
from __future__ import annotations

import collections
import difflib
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

RESOLUTION = 96            # ticks per quarter note after normalisation
NGRAM = 8
MIN_COSINE = 0.04
SPLIT_RELATIONS = {"highly_similar", "excerpt_or_structural_variant", "related_or_uncertain",
                   "melody_related_or_uncertain"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_id(prefix: str, members: list[str]) -> str:
    return prefix + digest("\n".join(sorted(members)).encode())[:16]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = collections.defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return list(out.values())


# ------------------------------------------------------------------ extraction
def normalize(notes) -> np.ndarray:
    a = np.asarray(notes, float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("empty or invalid score")
    a = a[np.isfinite(a).all(1) & (a[:, 0] >= 0) & (a[:, 0] <= 127) & (a[:, 2] > 0)]
    if len(a) < 2:
        raise ValueError("fewer than two score notes")
    a[:, 1] -= a[:, 1].min()
    a[:, 1:] = np.rint(a[:, 1:] * RESOLUTION)
    a[:, 2] = np.maximum(a[:, 2], 1)
    a = np.unique(a.astype(np.int64), axis=0)
    return a[np.lexsort((a[:, 2], a[:, 0], a[:, 1]))]


def outer_voices(a: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """(onset times, highest pitches) and (onset times, lowest pitches)."""
    times = np.unique(a[:, 1])
    idx = np.searchsorted(times, a[:, 1])
    lo, hi = np.full(len(times), 128), np.zeros(len(times), int)
    np.minimum.at(lo, idx, a[:, 0])
    np.maximum.at(hi, idx, a[:, 0])
    return [(times, hi), (times, lo)]


def extract(path: str) -> dict[str, Any]:
    """Normalised notes, hashes and outer voices of one score MIDI."""
    from symusic import Score
    try:
        midi = Score(str(path))
        tpq = midi.ticks_per_quarter
        a = normalize([[n.pitch, n.time / tpq, n.duration / tpq]
                       for t in midi.tracks if not t.is_drum for n in t.notes])
        transposed = a.copy()
        transposed[:, 0] -= transposed[0, 0]
        return {"path": str(path), "notes": a, "hash": digest(a.astype("<i8").tobytes()),
                "transposed_hash": digest(transposed.astype("<i8").tobytes()), "voices": outer_voices(a)}
    except Exception as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------- retrieval
def shingles(voices) -> set[tuple[int, ...]]:
    result = set()
    for _, pitches in voices:
        intervals = np.diff(pitches)
        for j in range(len(intervals) - NGRAM + 1):
            block = intervals[j:j + NGRAM]
            # Repeated notes / scales are weak evidence and not used as keys.
            if len(set(pitches[j:j + NGRAM + 1])) < 4 or len(set(block)) < 3:
                continue
            result.add(tuple(int(x) for x in block))
    return result


def retrieve(items: list[dict], min_cosine: float = MIN_COSINE) -> dict[tuple[int, int], float]:
    """All item pairs whose TF-IDF shingle cosine similarity is >= min_cosine."""
    df: collections.Counter = collections.Counter()
    sets = []
    for item in items:
        s = shingles(item["voices"])
        sets.append(s)
        df.update(s)
    keys = {s: i for i, s in enumerate(sorted(df))}
    rows, cols, data, n = [], [], [], len(items)
    for i, ss in enumerate(sets):
        for s in sorted(ss):
            rows.append(i)
            cols.append(keys[s])
            data.append(np.log((n + 1) / (df[s] + 1)) + 1)
    x = sparse.csr_matrix((data, (rows, cols)), shape=(n, len(keys)), dtype=np.float32)
    norm = np.sqrt(x.multiply(x).sum(1)).A1
    x = sparse.diags(1 / np.maximum(norm, 1e-12)) @ x
    pairs: dict[tuple[int, int], float] = {}
    for start in range(0, n, 128):
        sim = (x[start:start + 128] @ x.T).tocsr()
        for local in range(sim.shape[0]):
            i = start + local
            beg, end = sim.indptr[local:local + 2]
            for j, value in zip(sim.indices[beg:end], sim.data[beg:end]):
                if int(j) != i and value >= min_cosine:
                    pairs[tuple(sorted((i, int(j))))] = float(value)
    return pairs


# ---------------------------------------------------------------- verification
def verify(a: dict, b: dict) -> dict[str, Any]:
    """Anchored polyphonic verification of one candidate pair."""
    best = None
    for voice, ((ta, pa), (tb, pb)) in enumerate(zip(a["voices"], b["voices"])):
        blocks = [(x, y, n) for x, y, n in difflib.SequenceMatcher(
            None, np.diff(pa).tolist(), np.diff(pb).tolist(), autojunk=False).get_matching_blocks() if n >= 4]
        if not blocks:
            continue
        votes: collections.Counter = collections.Counter()
        for x, y, n in blocks:
            votes[int(pb[y] - pa[x])] += n
        shift = votes.most_common(1)[0][0]
        anchors = []
        for x, y, n in blocks:
            if int(pb[y] - pa[x]) != shift:
                continue
            da, db = np.diff(ta[x:x + n + 1]), np.diff(tb[y:y + n + 1])
            scale = float(np.median(db / da))
            if not 0.45 <= scale <= 2.2 or np.mean(np.abs(db - scale * da) / np.maximum(db, 1) <= 0.2) < 0.85:
                continue
            anchors.extend(zip(ta[x:x + n + 1], tb[y:y + n + 1]))
        if len(anchors) < 8:
            continue
        anchors = sorted(set(anchors))
        aa, bb = np.array([x for x, _ in anchors]), np.array([y for _, y in anchors])
        good = np.r_[True, (np.diff(aa) > 0) & (np.diff(bb) > 0)]
        aa, bb = aa[good], bb[good]
        if len(aa) < 8 or np.any(np.diff(aa) <= 0) or np.any(np.diff(bb) <= 0):
            continue
        # Map notes only inside the anchored range (no extrapolation).
        notes, other = a["notes"], b["notes"]
        by_pitch = {int(k): np.flatnonzero(other[:, 0] == k) for k in np.unique(other[:, 0])}
        predicted = np.interp(notes[:, 1], aa, bb)
        slopes = np.diff(bb) / np.diff(aa)
        scales = slopes[np.clip(np.searchsorted(aa, notes[:, 1], side="right") - 1, 0, len(slopes) - 1)]
        matched, used = [], set()
        for i, (pitch, onset, duration) in enumerate(notes):
            if onset < aa[0] or onset > aa[-1] or not 0.45 <= scales[i] <= 2.2:
                continue
            inds = by_pitch.get(int(pitch + shift))
            if inds is None:
                continue
            pos = np.searchsorted(other[inds, 1], predicted[i])
            options = []
            for j in inds[max(0, pos - 2):pos + 3]:
                if int(j) in used:
                    continue
                err = abs(other[j, 1] - predicted[i])  # tolerance: 1/32 quarter note
                dur_err = abs(other[j, 2] - duration * scales[i]) / max(other[j, 2], duration * scales[i], 1)
                if err <= 3 and dur_err <= 0.25:
                    options.append((err + dur_err, int(j)))
            if options:
                j = min(options)[1]
                used.add(j)
                matched.append((i, j))
        ca, cb = len(matched) / len(notes), len(matched) / len(other)
        coverage = []
        for v in (0, 1):
            va = set(zip(a["voices"][v][1].tolist(), a["voices"][v][0].tolist()))
            vb = set(zip(b["voices"][v][1].tolist(), b["voices"][v][0].tolist()))
            ma = sum((int(notes[i, 0]), int(notes[i, 1])) in va for i, _ in matched)
            mb = sum((int(other[j, 0]), int(other[j, 1])) in vb for _, j in matched)
            coverage.extend([min(ma / max(len(va), 1), 1), min(mb / max(len(vb), 1), 1)])
        span = min(aa[-1] - aa[0], bb[-1] - bb[0]) / RESOLUTION
        enough = len(matched) >= 48 and span >= 16 and len({notes[i, 0] for i, _ in matched}) >= 6
        relation = "local_similarity_or_different"
        if enough and min(ca, cb) >= 0.90 and min(coverage) >= 0.85:
            relation = "highly_similar"
        elif enough and max(ca, cb) >= 0.90 and min(ca, cb) >= 0.2:
            relation = "excerpt_or_structural_variant"
        elif enough and (max(ca, cb) >= 0.55 or max(coverage) >= 0.75):
            relation = "related_or_uncertain"
        result = {"relation": relation, "coverage_a": ca, "coverage_b": cb, "matched_notes": len(matched),
                  "transpose": shift, "anchor_voice": ["upper", "lower"][voice]}
        if best is None or len(matched) > best["matched_notes"]:
            best = result
    return best or {"relation": "insufficient_alignment_evidence", "coverage_a": 0, "coverage_b": 0, "matched_notes": 0}


def melody_relation(a: dict, b: dict) -> str:
    """Outer-voice interval comparison used as an extra split guard across corpora."""
    best = None
    for (ta, pa), (tb, pb) in zip(a["voices"], b["voices"]):
        ia, ib = np.diff(pa).tolist(), np.diff(pb).tolist()
        accepted, shifts = [], []
        for x, y, n in difflib.SequenceMatcher(None, ia, ib, autojunk=False).get_matching_blocks():
            if n < 8:
                continue
            da, db = np.diff(ta[x:x + n + 1]), np.diff(tb[y:y + n + 1])
            scale = np.median(db / np.maximum(da, 1))
            if not 0.45 <= scale <= 2.2 or (np.abs(db - scale * da) / np.maximum(db, 1) <= 0.25).mean() < 0.75:
                continue
            accepted.append((x, y, n))
            shifts.extend([int(pb[y] - pa[x])] * n)
        if not accepted:
            continue
        shift = collections.Counter(shifts).most_common(1)[0][0]
        accepted = [(x, y, n) for x, y, n in accepted if int(pb[y] - pa[x]) == shift]
        matched = sum(n for _, _, n in accepted)
        span = min(sum(ta[x + n] - ta[x] for x, _, n in accepted), sum(tb[y + n] - tb[y] for _, y, n in accepted)) / 96
        diversity = len({int(v) for x, _, n in accepted for v in pa[x:x + n + 1]})
        cand = (matched, matched / max(len(ia), 1), matched / max(len(ib), 1), span, diversity)
        if best is None or matched > best[0]:
            best = cand
    if best is None:
        return "rejected"
    matched, ca, cb, span, diversity = best
    short, long = max(ca, cb), min(ca, cb)
    if not (matched >= 48 and span >= 16 and diversity >= 6):
        return "rejected"
    if short >= 0.95 and matched >= 96:
        return "contained_work" if long < 0.60 else "near_work"
    if short >= 0.85 and long >= 0.60:
        return "near_work"
    return "uncertain" if short >= 0.55 else "rejected"


def verify_cross(a: dict, b: dict) -> dict[str, Any]:
    result = verify(a, b)
    melody = melody_relation(a, b)
    result["melody_relation"] = melody
    if result["relation"] not in SPLIT_RELATIONS and melody in ("uncertain", "near_work", "contained_work"):
        result["relation"] = "melody_related_or_uncertain"
    return result


# -------------------------------------------------------------------- grouping
def complete_link_groups(n: int, strong_edges: set[tuple[int, int]]) -> list[list[int]]:
    """Greedy deterministic partition where every member pair is directly linked."""
    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    for i, j in strong_edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    groups: list[list[int]] = []
    membership: dict[int, int] = {}
    for i in range(n):
        eligible = sorted({membership[j] for j in adjacency[i] if j in membership})
        dest = next((g for g in eligible if all(j in adjacency[i] for j in groups[g])), None)
        if dest is None:
            dest = len(groups)
            groups.append([])
        groups[dest].append(i)
        membership[i] = dest
    return groups
