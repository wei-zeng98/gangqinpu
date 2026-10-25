"""Filters 1 and 2: source audio duration and score instrumentation/length."""
from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

MIN_AUDIO_SECONDS = 60.0
MIN_BARS = 32
PIANO_TERMS = ("piano", "keyboard", "钢琴", "klavier", "fortepiano")


def audio_duration_seconds(audio: Path) -> float:
    """Container duration reported by ffprobe."""
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(audio)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "ffprobe could not read audio")
    return float(json.loads(completed.stdout)["format"]["duration"])


@dataclass(frozen=True)
class ScoreInspection:
    bar_count: int
    piano_only: bool
    parts: dict[str, str]


def _text(element: ET.Element | None) -> str:
    return " ".join(element.itertext()).lower() if element is not None else ""


def inspect_musicxml(path: Path) -> ScoreInspection:
    """Count piano measures and check that every sounding part is a piano.

    Only parts containing at least one non-rest note are considered sounding.
    Part labels (name, abbreviation, instrument) are matched against
    ``PIANO_TERMS``.
    """
    root = ET.parse(path).getroot()
    for element in root.iter():
        if "}" in element.tag:
            element.tag = element.tag.rsplit("}", 1)[-1]
    labels = {
        item.get("id", ""): " ".join(_text(child) for child in item)
        for item in root.findall(".//part-list/score-part")
    }
    sounding: dict[str, str] = {}
    for part in root.findall(".//part"):
        if any(note.find("rest") is None for note in part.findall(".//note")):
            part_id = part.get("id", "")
            sounding[part_id] = labels.get(part_id, "")
    if not sounding:
        raise ValueError("MusicXML contains no sounding parts")
    piano_only = all(any(term in label for term in PIANO_TERMS) for label in sounding.values())
    piano_bars = [
        len(part.findall("measure"))
        for part in root.findall(".//part")
        if any(term in sounding.get(part.get("id", ""), "") for term in PIANO_TERMS)
    ]
    return ScoreInspection(max(piano_bars, default=0), piano_only, sounding)
