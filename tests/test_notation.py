"""Semantic checks for general-purpose source notation and performance links."""
import copy
import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gangqinpu.notation import validate_notation, mark_beat_times
from test_hub import fixture
from gangqinpu.hub import native_record
from gangqinpu.dataset import read_metadata


class NotationTests(unittest.TestCase):
    def test_default_record_has_source_labels_without_xml_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);r=native_record(root,read_metadata(root)[0])
        self.assertNotIn('score.musicxml',r['files'])
        n=r['notation'];self.assertTrue(n['alignment_complete'])
        self.assertEqual(set(n),{'version','source_sha256','midi_origin_offset_quarter','parts','notes','measures','alignment_complete','issues'})
        self.assertEqual(n['version'],2)
        self.assertEqual(n['parts'],[{'id':'P1','staff_count':2}])
        self.assertEqual(n['notes'][0]['staff'],0);self.assertEqual(n['notes'][0]['voice'],0)
        self.assertFalse(n['notes'][0]['is_grace'])
        self.assertEqual(n['notes'][0]['performance_note_index'],0)
        self.assertEqual(n['measures'][0]['duration_quarter'],4.)
        self.assertEqual(set(n['notes'][0]),{'id','part_index','pitch','onset_quarter','duration_quarter','measure_index','staff','voice','is_grace','source_order','performance_note_index'})
        validate_notation(r,n)

    def test_third_staff_rest_is_preserved_without_rejecting_piece(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=fixture(root);x=p/'score.musicxml'
            x.write_text(x.read_text().replace('<staff>1</staff></note>\n</measure>', '<staff>3</staff></note>\n</measure>'))
            r=native_record(root,read_metadata(root)[0]);n=r['notation']
        self.assertEqual(n['parts'][0]['staff_count'],3)
        self.assertEqual(n['issues'],[]);self.assertTrue(n['alignment_complete'])
        self.assertEqual(len(n['notes']),1)

    def test_unknown_voice_is_nullable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);r=native_record(root,read_metadata(root)[0])
        r['notation']['notes'][0]['voice']=None
        validate_notation(r,r['notation'])
        self.assertIsNone(r['notation']['notes'][0]['voice'])

    def test_pedal_preserves_raw_midi_values(self):
        from symusic import Score,ControlChange
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=fixture(root);s=Score(str(p/'performance_clean.mid'))
            s.tracks[0].controls=[ControlChange(0,64,127),ControlChange(240,64,0)]
            s.dump_midi(str(p/'performance_clean.mid'));r=native_record(root,read_metadata(root)[0])
        self.assertEqual([e['value'] for e in r['performance_pedal']],[127,0])
        self.assertEqual(r['performance_pedal'][0]['time'],0.)
        self.assertGreater(r['performance_pedal'][1]['time'],0.)

    def test_absent_xml_voice_and_key_do_not_become_parser_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=fixture(root);x=p/'score.musicxml'
            x.write_text(x.read_text().replace('<voice>1</voice>','').replace('<key><fifths>0</fifths></key>',''))
            r=native_record(root,read_metadata(root)[0])
        self.assertIsNone(r['notation']['notes'][0]['voice'])
        self.assertIsNone(r['notation']['measures'][0]['key_fifths'])

    def test_corrupt_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);r=native_record(root,read_metadata(root)[0])
        r['notation']['notes'][0]['performance_note_index']=1
        with self.assertRaises(ValueError):validate_notation(r,r['notation'])

    def test_beat_flags_preserve_raw_order_and_low_confidence_downbeats(self):
        r={'audio_duration_seconds':3.,'beats':[{'performance_time_sec':v,'is_downbeat':True,'label':'bR','confidence':.25} for v in [0.,2.,1.,4.]]}
        mark_beat_times(r)
        self.assertEqual([b['time_valid'] for b in r['beats']],[True,False,False,False])
        self.assertEqual([b['performance_time_sec'] for b in r['beats']],[0.,2.,1.,4.])
        self.assertTrue(r['beats'][0]['is_downbeat']);self.assertEqual(r['beats'][0]['label'],'bR')

    def test_repeat_ids_are_unique_and_stable(self):
        from gangqinpu.pipeline.score import _repeat_playback_indices
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=fixture(root);x=p/'score.musicxml'
            s=x.read_text().replace('<attributes>','<barline location="left"><repeat direction="forward"/></barline><attributes>')
            s=s.replace('</measure>','<barline location="right"><repeat direction="backward"/></barline></measure>');x.write_text(s)
            tree,sequence=_repeat_playback_indices(x,unique_repeat_ids=True)
            names=[n.attrib['id'] for n in tree.findall('.//note') if 'id'in n.attrib]
        self.assertEqual(sequence,[0,0]);self.assertEqual(len(names),2);self.assertEqual(len(set(names)),2)
        self.assertTrue(any("__repeat_" in name for name in names))

    def test_multiple_parts_are_not_quality_errors(self):
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=fixture(root);x=p/'score.musicxml';tree=ET.parse(x);doc=tree.getroot()
            sp=copy.deepcopy(doc.find('./part-list/score-part'));sp.set('id','P2');doc.find('part-list').append(sp)
            part=copy.deepcopy(doc.find('part'));part.set('id','P2');part.find('.//note').set('id','n2');doc.append(part);tree.write(x)
            r=native_record(root,read_metadata(root)[0])
        self.assertEqual({n['part_index'] for n in r['notation']['notes']},{0,1})
        self.assertEqual(len(r['notation']['parts']),2);self.assertEqual(len(r['notation']['notes']),2)
        self.assertEqual(r['notation']['issues'],[])
        self.assertFalse(r['notation']['alignment_complete'])

    def test_source_grace_is_not_a_short_midi_note_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=fixture(root);x=p/'score.musicxml'
            x.write_text(x.read_text().replace('<note><rest/>','<note id="g1"><grace/><pitch><step>D</step><octave>4</octave></pitch><voice>1</voice><type>eighth</type><staff>1</staff></note><note><rest/>'))
            r=native_record(root,read_metadata(root)[0])
        graces=[n for n in r['notation']['notes'] if n['is_grace']]
        self.assertEqual(len(graces),1);self.assertEqual(graces[0]['duration_quarter'],0.)


if __name__=='__main__':unittest.main()
