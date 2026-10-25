"""MusicXML -> unfolded score note table, score MIDI (``score_MS.mid``) and beat grid."""
from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import mido
import numpy as np
import partitura as pt
from symupe.data.partitura import partitura_score_to_midi, preprocess_partitura_score
from .common import SCORE_MIDI, ScoreBeat, ScoreNote, as_int, sort_note_dicts


def _field(row: np.void, name: str, default: object = None) -> object:
    return row[name] if name in row.dtype.names else default


def _first_part(score: pt.score.Score) -> pt.score.Part:
    if not score.parts:
        raise ValueError("MusicXML score has no parts")
    return score.parts[0]


def _load_musicxml_with_sanitized_fallback(score_path: Path) -> pt.score.Score:
    """Load MusicXML, removing empty direction containers if partitura trips.

    Some converted MusicXML files contain empty ``<direction-type>``
    nodes. partitura 1.8.0 can raise ``StopIteration`` on those nodes before it
    reaches any note data. The score semantics are unaffected by deleting empty
    direction wrappers, so this fallback writes a temporary sanitized file and
    retries the parse.
    """
    try:
        return pt.load_musicxml(score_path, force_note_ids="keep")
    except StopIteration:
        tree = ET.parse(score_path)
        root = tree.getroot()
        removed = 0
        for parent in root.iter():
            for child in list(parent):
                tag = child.tag.split("}", 1)[-1]
                if (
                    tag == "direction-type"
                    and len(list(child)) == 0
                    and not (child.text or "").strip()
                ):
                    parent.remove(child)
                    removed += 1
        if removed == 0:
            raise
        with tempfile.NamedTemporaryFile(
            "wb", suffix=".musicxml", delete=False
        ) as handle:
            sanitized_path = Path(handle.name)
        try:
            tree.write(sanitized_path, encoding="utf-8", xml_declaration=True)
            return pt.load_musicxml(sanitized_path, force_note_ids="keep")
        finally:
            sanitized_path.unlink(missing_ok=True)


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _repeat_playback_indices(score_path: Path, *, unique_repeat_ids: bool = False) -> tuple[ET.ElementTree, list[int]]:
    """Return a repeat-free MusicXML tree and the explicit playback order.

    CCMZ exports first endings as a span followed by a backward repeat, but
    Partitura's generic repeat unroller can discard those repeats instead of
    taking the second pass.  Build the ordinary two-pass playback directly:
    play the full repeat once, then replay it while omitting its numbered-1
    endings.  This preserves written notation in ``score.musicxml`` while
    making score MIDI and alignment use the performed sequence.
    """
    tree = ET.parse(score_path)
    root = tree.getroot()
    parts = [child for child in root if _tag_name(child) == "part"]
    if not parts:
        return tree, []
    reference = [child for child in parts[0] if _tag_name(child) == "measure"]
    if not reference:
        return tree, []

    reserved_ids = {n.attrib['id'] for part in parts for n in part.iter()
                    if _tag_name(n) == 'note' and 'id' in n.attrib}
    if unique_repeat_ids:
        source_ids = [n.attrib['id'] for part in parts for n in part.iter()
                      if _tag_name(n) == 'note' and 'id' in n.attrib]
        if len(source_ids) != len(reserved_ids):
            raise ValueError('nonunique_source_note_ids')

    repeat_stack: list[int] = []
    repeat_pairs: list[tuple[int, int]] = []
    ending_starts: dict[str, list[int]] = {}
    ending_spans: list[tuple[str, int, int]] = []
    for index, measure in enumerate(reference):
        for barline in [child for child in measure if _tag_name(child) == "barline"]:
            for marker in barline:
                name = _tag_name(marker)
                if name == "repeat":
                    direction = marker.attrib.get("direction")
                    if direction == "forward":
                        repeat_stack.append(index)
                    elif direction == "backward" and repeat_stack:
                        repeat_pairs.append((repeat_stack.pop(), index))
                elif name == "ending":
                    number = marker.attrib.get("number", "")
                    ending_type = marker.attrib.get("type")
                    if ending_type == "start":
                        ending_starts.setdefault(number, []).append(index)
                    elif ending_type in {"stop", "discontinue"} and ending_starts.get(number):
                        ending_spans.append((number, ending_starts[number].pop(), index))

    if not repeat_pairs:
        return tree, list(range(len(reference)))

    sequence: list[int] = []
    cursor = 0
    for start, end in sorted(repeat_pairs, key=lambda pair: pair[1]):
        if start < cursor:
            # Nested/overlapping repeats are left to the notation parser;
            # CCMZ's ordinary adjacent repeats take the deterministic path.
            continue
        sequence.extend(range(cursor, end + 1))
        first_endings = [
            (ending_start, ending_end)
            for number, ending_start, ending_end in ending_spans
            if number == "1" and start <= ending_start <= ending_end <= end
        ]
        for measure_index in range(start, end + 1):
            if not any(ending_start <= measure_index <= ending_end for ending_start, ending_end in first_endings):
                sequence.append(measure_index)
        cursor = end + 1
    sequence.extend(range(cursor, len(reference)))

    for part in parts:
        measures = [child for child in part if _tag_name(child) == "measure"]
        if len(measures) != len(reference):
            return tree, list(range(len(reference)))
        expanded: list[ET.Element] = []
        visits = {}
        for measure_index in sequence:
            clone = deepcopy(measures[measure_index])
            visits[measure_index] = visits.get(measure_index, 0) + 1
            if unique_repeat_ids and visits[measure_index] > 1:
                # Identity changes only on copied playback occurrences. Never
                # merge grace notes or mutate the original MusicXML file.
                for note in clone.iter():
                    if _tag_name(note) != 'note' or 'id' not in note.attrib:
                        continue
                    stem = note.attrib['id'] + '__repeat_'
                    occurrence = visits[measure_index]
                    while stem + str(occurrence) in reserved_ids:
                        occurrence += 1
                    note.attrib['id'] = stem + str(occurrence)
                    reserved_ids.add(note.attrib['id'])
            for barline in [child for child in clone if _tag_name(child) == "barline"]:
                for marker in list(barline):
                    if _tag_name(marker) in {"repeat", "ending"}:
                        barline.remove(marker)
            expanded.append(clone)
        for measure in measures:
            part.remove(measure)
        part.extend(expanded)
    return tree, sequence


def _load_score_for_alignment(score_path: Path, *, unique_repeat_ids: bool = False) -> pt.score.Score:
    tree, sequence = _repeat_playback_indices(score_path, unique_repeat_ids=unique_repeat_ids)
    if not sequence:
        return _load_musicxml_with_sanitized_fallback(score_path)
    with tempfile.NamedTemporaryFile("wb", suffix=".musicxml", delete=False) as handle:
        unfolded_path = Path(handle.name)
    try:
        tree.write(unfolded_path, encoding="utf-8", xml_declaration=True)
        return _load_musicxml_with_sanitized_fallback(unfolded_path)
    finally:
        unfolded_path.unlink(missing_ok=True)


def _measure_for_div(part: pt.score.Part, onset_div: int) -> tuple[int, int, int]:
    try:
        measure_number = int(part.measure_number_map(onset_div))
        measure_start, measure_end = part.measure_map(onset_div)
        return measure_number, int(measure_start), int(measure_end)
    except Exception:
        return 1, 0, 0


def _beat_unit_quarters(ts_beats: int, ts_beat_type: int) -> float:
    # ASAP annotates beats in the metrical sense. For compound x/8 meters, the
    # natural beat is usually dotted-quarter; simple meters use the denominator.
    if ts_beat_type == 8 and ts_beats > 3 and ts_beats % 3 == 0:
        return 1.5
    return 4.0 / max(ts_beat_type, 1)


def _effective_time_signature(
    ts_beats: int,
    ts_beat_type: int,
    measure_start_div: int,
    measure_end_div: int,
    divs_pq: int,
) -> tuple[int, int, bool]:
    """Infer meter from an overfull MusicXML measure when it is unambiguous.

    Some converted scores keep a 4/4 declaration while a measure contains an
    exact extra beat of notation. Retaining 4/4 would make the beat grid call
    that fifth beat a stray beat and obscure the next downbeat. When the span
    is an integral number of denominator units, use that span as the local
    effective meter (for example, five quarters becomes 5/4).
    """
    denominator_unit_quarters = 4.0 / max(ts_beat_type, 1)
    declared_quarters = ts_beats * denominator_unit_quarters
    actual_quarters = (measure_end_div - measure_start_div) / max(divs_pq, 1)
    inferred_beats = round(actual_quarters / denominator_unit_quarters)
    if (
        actual_quarters > declared_quarters + 1e-7
        and inferred_beats > ts_beats
        and abs(actual_quarters - inferred_beats * denominator_unit_quarters) < 1e-7
    ):
        return inferred_beats, ts_beat_type, True
    return ts_beats, ts_beat_type, False


def _initial_pickup_offset_quarters(
    part: pt.score.Part,
    note_array: np.ndarray,
) -> float:
    """Return the missing duration before a genuine initial pickup.

    MusicXML represents an anacrusis as a short first measure even when its
    stated meter is, for example, 3/4. Its start is not a musical downbeat:
    the first full measure that follows is. The offset lets both notes and the
    beat grid describe that pickup as its final metrical beat(s).
    """
    if not part.measures:
        return 0.0

    first_measure = part.measures[0]
    start_div = as_int(first_measure.start.t)
    end_div = as_int(first_measure.end.t)
    if end_div <= start_div:
        return 0.0

    first_rows = [
        row for row in note_array if as_int(_field(row, "onset_div")) >= start_div
    ]
    if not first_rows:
        return 0.0
    first_row = first_rows[0]
    divs_pq = max(as_int(_field(first_row, "divs_pq"), 1), 1)
    ts_beats = as_int(_field(first_row, "ts_beats"), 4)
    ts_beat_type = as_int(_field(first_row, "ts_beat_type"), 4)
    expected_quarters = ts_beats * (4.0 / max(ts_beat_type, 1))
    actual_quarters = (end_div - start_div) / divs_pq
    # A short opening measure is a pickup. A longer opening span can be a
    # polymetric/irregular notation choice, so leave it untouched.
    if 1e-7 < actual_quarters < expected_quarters - 1e-7:
        return expected_quarters - actual_quarters
    return 0.0


def process_score(score_path: Path, output_dir: Path) -> dict:
    score_path = Path(score_path)
    output_dir = Path(output_dir)
    score_midi_path = output_dir / SCORE_MIDI

    # SyMuPe's converter is the canonical PianoCoRe-side preprocessing step.
    # We preprocess explicitly so the MusicXML table and score_MS.mid describe
    # the same unfolded playthrough.
    score = _load_score_for_alignment(score_path)
    score = preprocess_partitura_score(
        score,
        # `_load_score_for_alignment` has already expanded ordinary CCMZ
        # repeat/volta playback deterministically.
        unfold_repeats=False,
        unfold_minimal=False,
        expand_grace_notes=True,
        process_ornaments=True,
        clean_duplicates=True,
        cut_overlapped_notes=True,
    )
    partitura_score_to_midi(
        score,
        score_midi_path,
        unfold_repeats=False,
        expand_grace_notes=False,
        process_ornaments=False,
        clean_duplicates=False,
        cut_overlapped_notes=False,
        clean_short_notes=True,
        ticks_per_quarter=480,
    )

    note_array = score.note_array(
        include_time_signature=True,
        include_metrical_position=True,
        include_pitch_spelling=True,
        include_key_signature=True,
        include_staff=True,
        include_divs_per_quarter=True,
    )
    part = _first_part(score)
    playthrough_starts = [as_int(measure.start.t) for measure in part.measures]
    pickup_offset_quarters = _initial_pickup_offset_quarters(part, note_array)
    first_measure_start_div = as_int(part.measures[0].start.t) if part.measures else 0

    raw_notes: list[ScoreNote] = []
    for row in note_array:
        pitch = as_int(_field(row, "pitch"))
        onset_div = as_int(_field(row, "onset_div"))
        duration_div = as_int(_field(row, "duration_div"))
        divs_pq = max(as_int(_field(row, "divs_pq"), 1), 1)
        # SyMuPe serializes an initial pickup at the end of a virtual full bar
        # in score_MS.mid. Partitura's onset_quarter stays pickup-normalized,
        # while onset_div remains in source divisions, so derive the score
        # coordinate from divisions plus the same pickup padding as the MIDI.
        onset_quarter = onset_div / divs_pq + pickup_offset_quarters
        duration_quarter = duration_div / divs_pq
        declared_ts_beats = as_int(_field(row, "ts_beats"), 4)
        declared_ts_beat_type = as_int(_field(row, "ts_beat_type"), 4)
        measure, measure_start_div, measure_end_div = _measure_for_div(part, onset_div)
        ts_beats, ts_beat_type, time_signature_inferred = _effective_time_signature(
            declared_ts_beats,
            declared_ts_beat_type,
            measure_start_div,
            measure_end_div,
            divs_pq,
        )
        playthrough_measure_index = max(
            1,
            sum(start <= onset_div for start in playthrough_starts),
        )
        measure_start_quarter = measure_start_div / divs_pq + pickup_offset_quarters
        beat_unit = _beat_unit_quarters(ts_beats, ts_beat_type)
        pickup_offset = (
            pickup_offset_quarters
            if measure_start_div == first_measure_start_div and onset_div >= measure_start_div
            else 0.0
        )
        beat_in_measure = 1.0 + ((onset_quarter - measure_start_quarter + pickup_offset) / beat_unit)
        # Assign score indices after sorting by onset, pitch, and duration.
        raw_notes.append(
            {
                "pitch": pitch,
                "onset_quarter": onset_quarter,
                "duration_quarter": duration_quarter,
                "measure": measure,
                "source_measure": measure,
                "playthrough_measure_index": playthrough_measure_index,
                "beat_in_measure": beat_in_measure,
                "staff": as_int(_field(row, "staff"), -1),
                "voice": as_int(_field(row, "voice"), -1),
                "id": str(_field(row, "id", "")),
                "onset_div": onset_div,
                "duration_div": duration_div,
                "divs_pq": divs_pq,
                "time_signature": f"{ts_beats}/{ts_beat_type}",
                "time_signature_inferred": time_signature_inferred,
                "key_signature": str(_field(row, "ks_fifths", "")),
                "key_fifths": as_int(_field(row, "ks_fifths")),
                "key_mode": as_int(_field(row, "ks_mode")),
            }
        )

    score_notes = sort_note_dicts(raw_notes, "onset_quarter", "duration_quarter")
    for index, note in enumerate(score_notes):
        note["score_note_index"] = index

    score_beats = _build_score_beats(part, score_notes, pickup_offset_quarters)
    origin_offset = score_origin_offset(score_midi_path, score_notes)
    score_times = midi_quarter_to_seconds(
        score_midi_path, [beat["onset_quarter"] - origin_offset for beat in score_beats]
    )
    for beat, time_sec in zip(score_beats, score_times):
        beat["score_time_sec"] = time_sec

    return {
        "score_midi_path": str(score_midi_path),
        "score_notes": score_notes,
        "score_beats": score_beats,
    }


def _build_score_beats(
    part: pt.score.Part,
    score_notes: list[ScoreNote],
    pickup_offset_quarters: float,
) -> list[ScoreBeat]:
    """Create an ASAP-style score beat grid from the unfolded partitura score."""
    beats: list[ScoreBeat] = []
    beat_index = 0
    last_ts = "4/4"
    last_beat_unit = 1.0
    last_key_fifths = 0
    last_key_mode = 0
    for playthrough_measure_index, measure_obj in enumerate(part.measures, start=1):
        measure_number = as_int(getattr(measure_obj, "number", None), len(beats) + 1)
        measure_start_div = as_int(measure_obj.start.t)
        measure_end_div = as_int(measure_obj.end.t)
        measure_notes = [
            note
            for note in score_notes
            if measure_start_div <= as_int(note["onset_div"]) < measure_end_div
        ]
        if measure_notes:
            first_note = measure_notes[0]
            last_ts = str(first_note.get("time_signature", last_ts))
            last_key_fifths = as_int(first_note.get("key_fifths"), last_key_fifths)
            last_key_mode = as_int(first_note.get("key_mode"), last_key_mode)
            ts_beats, ts_beat_type = [as_int(part) for part in last_ts.split("/", 1)]
            last_beat_unit = _beat_unit_quarters(ts_beats, ts_beat_type)
            divs_pq = max(as_int(first_note.get("divs_pq"), 1), 1)
            time_signature_inferred = bool(first_note.get("time_signature_inferred", False))
        else:
            ts_beats, ts_beat_type = [as_int(part) for part in last_ts.split("/", 1)]
            divs_pq = max(as_int(getattr(measure_obj.start, "quarter", None), 24), 1)
            time_signature_inferred = False

        # Keep this grid in score_MS.mid coordinates. This is especially
        # important for a pickup, which SyMuPe pads before serializing MIDI.
        start_quarter = as_int(measure_obj.start.t) / divs_pq + pickup_offset_quarters
        end_quarter = as_int(measure_obj.end.t) / divs_pq + pickup_offset_quarters
        # Do not invent a complete bar when a score has an empty terminal
        # measure. This avoids squeezing non-existent end beats onto the final
        # aligned onset.
        if end_quarter <= start_quarter + 1e-7:
            continue

        pos = start_quarter
        is_pickup = playthrough_measure_index == 1 and pickup_offset_quarters > 1e-7
        beat_in_measure = 1.0 + (pickup_offset_quarters / last_beat_unit if is_pickup else 0.0)
        while pos < end_quarter - 1e-7:
            beats.append(
                {
                    "score_beat_index": beat_index,
                    "onset_quarter": float(pos),
                    "measure": measure_number,
                    "beat_in_measure": float(beat_in_measure),
                    "is_downbeat": beat_in_measure == 1.0 and not is_pickup,
                    "time_signature": last_ts,
                    "key_fifths": last_key_fifths,
                    "key_mode": last_key_mode,
                    "is_pickup": is_pickup,
                    "time_signature_inferred": time_signature_inferred,
                }
            )
            beat_index += 1
            beat_in_measure += 1.0
            pos += last_beat_unit
    return beats


def score_origin_offset(midi_path: Path, score_notes: list[ScoreNote]) -> float:
    """Construction-score quarter origin relative to the serialized score MIDI."""
    from symusic import Score
    midi = Score(str(midi_path))
    notes = sorted([n for track in midi.tracks if not track.is_drum for n in track.notes],
                   key=lambda n: (n.time, n.pitch, n.duration))
    if len(notes) != len(score_notes) or any(n.pitch != s["pitch"] for n, s in zip(notes, score_notes)):
        raise RuntimeError("score MIDI does not preserve the score-table note bridge")
    offsets = np.asarray([s["onset_quarter"] - n.time / midi.ticks_per_quarter
                          for n, s in zip(notes, score_notes)])
    if not len(offsets):
        return 0.0
    tolerance = 1 / midi.ticks_per_quarter + 1e-8
    if np.ptp(offsets) > tolerance:
        raise RuntimeError("score-table and MIDI quarter coordinates differ non-uniformly")
    offset = float(np.median(offsets))
    return 0.0 if abs(offset) <= tolerance else offset


def midi_quarter_to_seconds(midi_path: Path, quarter_positions: list[float]) -> list[float]:
    """Map MIDI quarter positions through explicit tempo events, preserving order."""
    midi = mido.MidiFile(midi_path)
    # An explicit event at tick zero replaces the MIDI default; it must not
    # compete with a synthetic default after sorting by tempo value.
    events = {0: 500000}
    tick = 0
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.type == "set_tempo":
            events[tick] = message.tempo
    ticks = np.asarray(sorted(events), dtype=np.int64)
    tempi = np.asarray([events[int(t)] for t in ticks], dtype=float)
    seconds = np.r_[0.0, np.cumsum(np.diff(ticks) * tempi[:-1] / (1e6 * midi.ticks_per_beat))]
    targets = np.rint(np.asarray(quarter_positions, dtype=float) * midi.ticks_per_beat).astype(np.int64)
    indices = np.maximum(0, np.searchsorted(ticks, targets, side="right") - 1)
    return (seconds[indices] + (targets - ticks[indices]) * tempi[indices]
            / (1e6 * midi.ticks_per_beat)).tolist()
