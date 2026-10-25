"""Compact, XML-free notation targets. Extraction is a build-time operation.

Indices address lists in the same record. Quarter positions use the construction
score clock, as do beats; MIDI positions differ by ``midi_origin_offset_quarter``.
Unknown labels/links are null, never guessed from pitch or note duration.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def notation_features():
    from datasets import List, Value
    return {
        "version": Value("int16"), "source_sha256": Value("string"),
        "midi_origin_offset_quarter": Value("float64"),
        "notes": List({
            "id": Value("string"), "part_index": Value("int16"), "pitch": Value("int16"),
            "onset_quarter": Value("float64"), "duration_quarter": Value("float64"),
            "measure_index": Value("int32"), "staff": Value("int16"), "voice": Value("int16"),
            "is_grace": Value("bool"), "source_order": Value("int32"),
            "performance_note_index": Value("int32"),
        }),
        "measures": List({
            "part_index": Value("int16"),
            "onset_quarter": Value("float64"), "duration_quarter": Value("float64"),
            "time_signature": Value("string"), "time_signature_inferred": Value("bool"),
            "key_fifths": Value("int16"), "is_pickup": Value("bool"),
        }),
        "parts": List({"id": Value("string"), "staff_count": Value("int16")}),
        "alignment_complete": Value("bool"), "issues": List(Value("string")),
    }


def mark_beat_times(record):
    """Annotate, never drop/reorder/clip raw reference beats.

    Both endpoints of a non-increasing adjacent pair are marked invalid.
    Confidence remains a separate heuristic weight, including for downbeats.
    """
    beats = record["beats"]
    times = np.array([b["performance_time_sec"] for b in beats], dtype=float)
    duration = record.get("audio_duration_seconds")
    valid = np.isfinite(times) & (times >= 0)
    if duration is not None:
        valid &= times <= duration
    bad = np.flatnonzero(np.diff(times) <= 0)
    valid[bad] = False; valid[bad+1] = False
    for beat, value in zip(beats, valid):
        beat["time_valid"] = bool(value)


def pedal_records(sample_dir):
    """Track-0 CC64 events on the performance clock, with raw MIDI values."""
    from symusic import Score
    score=Score(str(Path(sample_dir)/"performance_clean.mid")).to("second")
    if not score.tracks:return []
    a=score.tracks[0].controls.numpy()
    return [{"time":float(t),"value":int(v)} for t,n,v in zip(a["time"],a["number"],a["value"]) if n==64]


def source_parts(xml):
    """Inventory all declared staves, including rests and inactive source rows."""
    root=ET.parse(xml).getroot()
    for node in root.iter():node.tag=node.tag.rsplit('}',1)[-1]
    parts=[]
    for index,part in enumerate(root.findall('part')):
        staff=[int(n.findtext('staff','1')) for n in part.findall('./measure/note')]
        declared=[int(n.text) for n in part.findall('./measure/attributes/staves') if n.text]
        parts.append({'id':part.attrib.get('id',str(index)), 'staff_count':max([1,*staff,*declared])})
    return parts


def source_label_presence(xml):
    """Avoid turning a parser's default C major/voice 1 into a source label."""
    root=ET.parse(xml).getroot()
    for n in root.iter():n.tag=n.tag.rsplit('}',1)[-1]
    pitched=[n for n in root.findall('./part/measure/note') if n.find('pitch') is not None]
    voice=bool(pitched) and all((n.findtext('voice') or '').strip() for n in pitched)
    return voice, bool(root.findall('.//key/fifths')), bool(root.findall('.//time'))


def midi_origin_offset(record, sample_dir):
    """Recover the release's constant offset with pitch/index witnesses."""
    with np.load(Path(sample_dir) / "performance_align.npz", allow_pickle=False) as raw:
        ids = np.asarray(raw["matched_score_indices"], dtype=int)
        notes = record["score_notes"]
        if not len(ids) or (ids < 0).any() or (ids >= len(notes)).any():
            raise ValueError("cannot_verify_score_origin_indices")
        pitch = np.array([notes[i]["pitch"] for i in ids])
        if not np.array_equal(pitch, raw["matched_score_pitches"]):
            raise ValueError("cannot_verify_score_origin_pitches")
        delta = raw["matched_score_onsets"] - np.array([notes[i]["onset_quarter"] for i in ids])
        offset = float(np.median(delta))
        if not np.isfinite(delta).all() or np.max(np.abs(delta - offset)) > 1. / 480 + 1e-7:
            raise ValueError("nonconstant_score_origin")
        return round(offset * 480) / 480


def _measures(starts, lengths, meters, inferred, keys, beats):
    pickup = {int(b["measure"]) for b in beats if b["is_pickup"]}
    # Beat measure labels are one-based unfolded labels in the released tables.
    return [dict(part_index=0, onset_quarter=float(q), duration_quarter=float(d), time_signature=str(ts),
                 time_signature_inferred=bool(inf), key_fifths=None if key is None else int(key),
                 is_pickup=(i + 1 in pickup))
            for i, (q, d, ts, inf, key) in enumerate(zip(starts, lengths, meters, inferred, keys))]


def _finish(record, xml, notes, measures, offset, issues=()):
    """Attach source facts and diagnostics without imposing a model format."""
    issues=list(issues)
    if any(n['duration_quarter'] is None for n in notes):issues.append('unknown_note_duration')
    if any(n['staff'] is None for n in notes):issues.append('unknown_staff')
    if not measures:issues.append('missing_measures')
    if any(not n['is_grace'] and n['duration_quarter']==0 for n in notes):issues.append('zero_duration_non_grace')
    if any(m['duration_quarter']<=0 for m in measures):issues.append('zero_length_measure')
    for n in notes:
        m=measures[n['measure_index']]
        if not m['onset_quarter']-1e-7 <= n['onset_quarter'] < m['onset_quarter']+m['duration_quarter']-1e-9:
            issues.append('note_outside_measure')
    mapped=[n['performance_note_index'] for n in notes if n['performance_note_index'] is not None]
    complete=sorted(mapped)==list(range(len(record['performance_notes'])))
    out=dict(version=2,source_sha256=hashlib.sha256(Path(xml).read_bytes()).hexdigest(),
             midi_origin_offset_quarter=offset,parts=source_parts(xml),notes=notes,measures=measures,
             alignment_complete=complete,issues=sorted(set(issues)))
    validate_notation(record,out)
    return out


def extract_notation(record, xml):
    """Extract source notation without grace/ornament realization; link conservatively."""
    from .pipeline.score import _load_score_for_alignment, _initial_pickup_offset_quarters
    from symupe.data.partitura import preprocess_partitura_score
    score = _load_score_for_alignment(Path(xml), unique_repeat_ids=True)
    if len(score.parts) != 1:
        return _extract_multipart(record, xml, score)
    part = score.parts[0]
    explicit_voice,explicit_key,explicit_meter=source_label_presence(xml)
    raw = score.note_array(include_staff=True, include_divs_per_quarter=True,
                           include_grace_notes=True, include_time_signature=True).copy()
    ids = list(map(str, raw["id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("nonunique_source_note_ids")
    objects = {str(n.id): n for n in part.notes_tied}
    padding = _initial_pickup_offset_quarters(part, raw)
    onsets = raw["onset_div"]/raw["divs_pq"] + padding
    durations = raw["duration_div"]/raw["divs_pq"]
    grace = raw["is_grace"].astype(bool)
    durations[grace] = 0
    order = np.lexsort((np.arange(len(raw)), raw["local_grace_order"], ~grace,
                       raw["voice"], raw["staff"], onsets))
    rank = np.empty(len(raw), dtype=int); rank[order] = np.arange(len(raw))
    voices = {str(v): i for i,v in enumerate(sorted(set(raw["voice"][raw["voice"] >= 0])))}
    starts = np.array([float(part.quarter_map(m.start.t)) + padding for m in part.measures])
    ends = np.array([float(part.quarter_map(m.end.t)) + padding for m in part.measures])
    # Represent the virtual initial pickup padding explicitly as part of bar 0.
    starts[0] -= padding
    meters = []; keys = []; inferred = []
    for measure in part.measures:
        ts = part.time_signature_map(measure.start.t)
        meters.append(f"{int(ts[0])}/{int(ts[1])}")
        key = part.key_signature_map(measure.start.t)
        keys.append(int(key[0]) if explicit_key and np.isfinite(key[0]) else None)
        inferred.append(not explicit_meter)
    offset = midi_origin_offset(record, Path(xml).parent)
    source_lookup = {name:i for i,name in enumerate(ids)}
    realized = preprocess_partitura_score(score, unfold_repeats=False, unfold_minimal=False)
    canonical = realized.note_array(include_staff=True, include_divs_per_quarter=True, include_time_signature=True)
    pad = _initial_pickup_offset_quarters(realized.parts[0], canonical)
    candidates = defaultdict(list)
    for row in canonical:
        key = (int(row["pitch"]), round((row["onset_div"]/row["divs_pq"]+pad-offset)*480),
               round(row["duration_div"]/row["divs_pq"]*480))
        candidates[key].append(source_lookup.get(str(row["id"])))
    linked = defaultdict(list)
    for i,n in enumerate(record["score_notes"]):
        key = (n["pitch"],round(n["onset_quarter"]*480),round(n["duration_quarter"]*480))
        choices = candidates.get(key, [])
        p = record["alignment"]["performance_note_indices"][i]
        if p >= 0 and len(choices) == 1 and choices[0] is not None:
            linked[choices[0]].append(p)
    notes = []
    for i,name in enumerate(ids):
        ps = linked.get(i, [])
        notes.append(dict(id=name, part_index=0, pitch=int(raw["pitch"][i]), onset_quarter=float(onsets[i]),
                          duration_quarter=float(durations[i]) if durations[i]>=0 else None, measure_index=int(np.searchsorted(starts,onsets[i],side="right")-1),
                          staff=int(raw["staff"][i])-1 if raw["staff"][i]>0 else None,
                          voice=voices.get(str(raw["voice"][i])) if explicit_voice else None, is_grace=bool(grace[i]), source_order=int(rank[i]),
                          performance_note_index=ps[0] if len(ps)==1 else None,
                          ))
    issues = []
    if any(not getattr(n,"is_printed",True) for n in objects.values()):
        issues.append("hidden_realization_notes")
    return _finish(record, xml, notes, _measures(starts, ends-starts, meters,inferred,keys,record["beats"]),offset,issues)


def _extract_multipart(record, xml, score):
    """Preserve every source part with a shared score-time origin."""
    from .pipeline.score import _initial_pickup_offset_quarters
    notes=[];measures=[]
    first_raw=score.parts[0].note_array(include_staff=True,include_divs_per_quarter=True,
                                       include_grace_notes=True,include_time_signature=True)
    padding=_initial_pickup_offset_quarters(score.parts[0],first_raw)
    source=ET.parse(xml).getroot()
    for n in source.iter():n.tag=n.tag.rsplit('}',1)[-1]
    source_parts=source.findall('part')
    for pi,part in enumerate(score.parts):
        raw=part.note_array(include_staff=True,include_divs_per_quarter=True,
                            include_grace_notes=True,include_time_signature=True)
        sp=source_parts[pi]
        pitched=[n for n in sp.findall('./measure/note') if n.find('pitch') is not None]
        explicit_voice=bool(pitched) and all((n.findtext('voice') or '').strip() for n in pitched)
        explicit_key=bool(sp.findall('.//key/fifths'))
        explicit_meter=bool(sp.findall('.//time'))
        base=len(measures)
        starts=np.array([float(part.quarter_map(m.start.t))+padding for m in part.measures])
        ends=np.array([float(part.quarter_map(m.end.t))+padding for m in part.measures])
        starts[0]-=padding
        for j,m in enumerate(part.measures):
            ts=part.time_signature_map(m.start.t);key=part.key_signature_map(m.start.t)
            measures.append(dict(part_index=pi,onset_quarter=float(starts[j]),duration_quarter=float(ends[j]-starts[j]),
                                 time_signature=f"{int(ts[0])}/{int(ts[1])}",time_signature_inferred=not explicit_meter,
                                 key_fifths=int(key[0]) if explicit_key and np.isfinite(key[0]) else None,is_pickup=(j==0 and padding>0)))
        voices={int(v):i for i,v in enumerate(sorted(set(raw['voice'][raw['voice']>=0])))}
        for row in raw:
            q=float(row['onset_div']/row['divs_pq']+padding)
            notes.append(dict(id=f"part{pi}:{row['id']}",part_index=pi,pitch=int(row['pitch']),
                              onset_quarter=q,duration_quarter=0. if row['is_grace'] else (float(row['duration_div']/row['divs_pq']) if row['duration_div']>=0 else None),
                              measure_index=base+int(np.searchsorted(starts,q,side='right')-1),
                              staff=int(row['staff'])-1 if row['staff']>0 else None,voice=voices.get(int(row['voice'])) if explicit_voice else None,
                              is_grace=bool(row['is_grace']),source_order=len(notes),performance_note_index=None,
                              ))
    return _finish(record,xml,notes,measures,midi_origin_offset(record,Path(xml).parent))


def validate_notation(record, notation):
    notes, measures = notation["notes"], notation["measures"]
    if len({n["id"] for n in notes}) != len(notes):
        raise ValueError("duplicate_notation_id")
    if sorted(n["source_order"] for n in notes) != list(range(len(notes))):
        raise ValueError("invalid_source_order")
    previous = defaultdict(lambda: -np.inf)
    for m in measures:
        q,d = m["onset_quarter"],m["duration_quarter"]
        part=m.get("part_index",0)
        if not np.isfinite([q,d]).all() or q < previous[part] or d < 0:
            raise ValueError("invalid_measure_geometry")
        previous[part] = q
    mapped = []
    for i,n in enumerate(notes):
        q,d = n["onset_quarter"],n["duration_quarter"]
        if not np.isfinite(q) or (d is not None and (not np.isfinite(d) or d < 0)) or (n["is_grace"] and d != 0):
            raise ValueError("invalid_notation_duration")
        if d is None and 'unknown_note_duration' not in notation['issues']:
            raise ValueError('unknown_duration_requires_diagnostic')
        bar = n["measure_index"]
        if not 0 <= bar < len(measures):
            raise ValueError("invalid_measure_index")
        m = measures[bar]
        part=n.get('part_index',0)
        if not 0<=part<len(notation['parts']):
            raise ValueError('invalid_part_index')
        if n['staff'] is not None and not 0<=n['staff']<notation['parts'][part]['staff_count']:
            raise ValueError('invalid_staff_index')
        if n.get("part_index",0) != m.get("part_index",0):
            raise ValueError("note_measure_part_mismatch")
        p = n["performance_note_index"]
        if p is not None:
            if not 0 <= p < len(record["performance_notes"]) or record["performance_notes"][p]["pitch"] != n["pitch"]:
                raise ValueError("invalid_notation_performance_link")
            mapped.append(p)
    if len(mapped) != len(set(mapped)):
        raise ValueError("duplicate_notation_performance_link")
    complete=sorted(mapped)==list(range(len(record['performance_notes'])))
    if notation['alignment_complete']!=complete:
        raise ValueError('inconsistent_alignment_coverage')
