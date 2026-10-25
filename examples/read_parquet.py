#!/usr/bin/env python3
"""Inspect structured scores and performances from local default Parquet."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    files=sorted(str(f) for f in (args.root/'data').glob('train-*.parquet'))
    if not files:parser.error('no training split shards found')
    sample=next(iter(load_dataset('parquet',data_files={'train':files},split='train',streaming=True)))
    n=sample['notation']
    print({'sample_id':sample['sample_id'],'parts':n['parts'],'performance_notes':len(sample['performance_notes']),
           'notation_notes':len(n['notes']),'measures':len(n['measures']),'pedal_events':len(sample['performance_pedal']),
           'alignment_complete':n['alignment_complete'],'issues':n['issues']})


if __name__=='__main__':main()
