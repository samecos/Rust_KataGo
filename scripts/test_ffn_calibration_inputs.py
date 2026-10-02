"""Bounded CPU tests; reads frozen input metadata/tensors only, never NN outputs."""
import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import ffn_calibration_inputs as m

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'target/unified-quant-calibration-encoding-real-r2/complete2044'
EXPECTED={1:[507,369,1530,736,896,22,1878,372],3:[2672,2483,2328,2074,2264,2271,2122,2367],8:[2869,2780,2731,2942,2962,2861,2908,2969]}
SHAS={'b11':'f51a70490737f37488c218a94f34d61e1bde73834f82b5825c27cd96f5dcd19d','b15':'e64881299585c8aaaa16a5b1b662b36292b5c39463533963324d5a6aab0d3e37'}


class Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded={};cls.sources={};cls.anchors={}
        for family in ('b11','b15'):
            sources=m.Sources();anchor=sources.anchor(BASE/family/'encoded/provenance.json',SHAS[family])
            cls.loaded[family]=m.load_inputs(anchor,sources)
            cls.sources[family]=sources;cls.anchors[family]=anchor

    def test_actual_complete_source_order_and_exact_tails(self):
        for family,inputs in self.loaded.items():
            self.assertEqual(len(inputs),2982)
            self.assertEqual([x['index'] for x in inputs[:2044]],list(range(2044)))
            self.assertEqual([x['rows'][0]['logical_row'] for x in inputs[:2044]],list(range(2044)))
            self.assertEqual((inputs[2725]['target_batch'],inputs[2725]['physical_batch']),(3,1))
            self.assertEqual((inputs[2981]['target_batch'],inputs[2981]['physical_batch']),(8,4))
            for B in (1,3,8):
                rows=[r['logical_row'] for x in inputs if x['target_batch']==B for r in x['rows']]
                self.assertEqual(rows,list(range(2044)))
            self.assertEqual(len(self.sources[family].items),1+1+1+1+8+2044+24+2*2982)
            self.sources[family].recheck()

    def test_actual_hash_ranked_inputs_are_model_independent(self):
        self.assertEqual([(x['id'],x['tensor_sha256']) for x in self.loaded['b11']],[(x['id'],x['tensor_sha256']) for x in self.loaded['b15']])
        for values in self.loaded.values():
            costs=m.choose_cost_inputs(values)
            self.assertEqual({x['physical_batch']:x['input_indices'] for x in costs},EXPECTED)
            self.assertTrue(all(x['warmup']==3 and x['measurements']==4 for x in costs))
            self.assertEqual(2044+sum(x['warmup']+len(x['input_indices'])*(x['measurements']+2) for x in costs),2197)
            self.assertEqual(2044+sum(x['physical_batch']*(x['warmup']+len(x['input_indices'])*(x['measurements']+2)) for x in costs),2656)
            self.assertEqual(2197*80,175760);self.assertEqual(2656*80,212480)

    def test_model_specific_pb_binding_and_row_identity_preserved(self):
        a=self.loaded['b11'][0]['rows'][0];b=self.loaded['b15'][0]['rows'][0]
        self.assertEqual(a['row_feature_sha256'],b['row_feature_sha256'])
        self.assertNotEqual(a['pb_sha256'],b['pb_sha256'])
        self.assertEqual(a['pb']['sha256'],a['pb_sha256'])
        self.assertEqual(a['source_line'],a['task_id']);self.assertEqual(a['logical_row']+1,a['source_line'])
        for key in ('game_id','semantic_position_sha256','board_state_sha256','wire_semantic_sha256','source_line_sha256'):
            self.assertTrue(m.valid_sha(a[key]))

    def test_prefix_partial_or_wrong_packing_is_not_complete(self):
        p=self.sources['b11'].json(self.anchors['b11'])
        for key,value in [('coverage_mode','prefix-diagnostic'),('encoded_count',9),('packings',[1,3]),('source_split','holdout'),('source_count',True)]:
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):m._provenance(bad)

    def test_wrong_provenance_sha_is_rejected_before_reading_tree(self):
        source=dict(self.anchors['b11'],sha256='0'*64)
        with self.assertRaises(ValueError):m.load_inputs(source,m.Sources())

    def test_row_offset_swap_task_or_padding_rejected(self):
        d=self.loaded['b11'][2044];root=BASE/'b11/encoded'
        raw=(root/'rows.jsonl').read_bytes()
        row=next(m.decode_json(x) for x in raw.splitlines() if m.decode_json(x)['input_id']==d['id'] and m.decode_json(x)['row']==1)
        s=dict(d['spatial'],path=row['spatial_file']);g=dict(d['global'],path=row['global_file'])
        m._row_map(row,d,1,row['manifest'],s,g)
        for key,value in [('spatial_offset_bytes',0),('logical_row',0),('physical_batch',1),('padding',True),('task_id',1),('row',True)]:
            bad=dict(row);bad[key]=value
            with self.assertRaises(ValueError):m._row_map(bad,d,1,row['manifest'],s,g)

    def test_tensor_reordered_or_nonfinite_cannot_pass_rebound_batch_hash(self):
        spatial=struct.pack('<f',.25)*(22*361)+struct.pack('<f',.5)*(22*361)
        global_values=struct.pack('<f',1.)*19+struct.pack('<f',2.)*19
        proofs=[];rows=[]
        for i in (0,1):
            s=spatial[i*m.SPATIAL_BYTES:(i+1)*m.SPATIAL_BYTES];g=global_values[i*m.GLOBAL_BYTES:(i+1)*m.GLOBAL_BYTES]
            feature=m.sha(m.ROW_DOMAIN+s+g)
            proofs.append(dict(post_spatial_sha256=m.sha(s),global_sha256=m.sha(g),post_feature_sha256=feature))
            rows.append(dict(logical_row=i,row_feature_sha256=feature))
        d=dict(physical_batch=2,rows=rows,tensor_sha256=m.sha(m.TENSOR_DOMAIN+struct.pack('<Q',2)+spatial+global_values))
        m._tensor_rows(d,spatial,global_values,proofs)
        swapped=spatial[m.SPATIAL_BYTES:]+spatial[:m.SPATIAL_BYTES]
        d['tensor_sha256']=m.sha(m.TENSOR_DOMAIN+struct.pack('<Q',2)+swapped+global_values)
        with self.assertRaises(ValueError):m._tensor_rows(d,swapped,global_values,proofs)
        nan=struct.pack('<I',0x7fc00000)+spatial[4:]
        d['tensor_sha256']=m.sha(m.TENSOR_DOMAIN+struct.pack('<Q',2)+nan+global_values)
        with self.assertRaises(ValueError):m._tensor_rows(d,nan,global_values,proofs)

    def test_rank_input_duplicates_reorder_and_insufficient_full_batch_rejected(self):
        original=self.loaded['b11']
        duplicate=copy.deepcopy(original);duplicate[1]['id']=duplicate[0]['id']
        with self.assertRaises(ValueError):m.choose_cost_inputs(duplicate)
        with self.assertRaises(ValueError):m.choose_cost_inputs(list(reversed(original)))
        tiny=copy.deepcopy(original[:2044]+original[2044:2052]+original[2975:])
        for i,item in enumerate(tiny):item['index']=i
        with self.assertRaises(ValueError):m.choose_cost_inputs(tiny)

    def test_jsonl_and_relative_bindings_reject_ambiguous_identity(self):
        self.assertEqual(m._lines(b'one\r\ntwo\n'),[b'one',b'two'])
        for raw in (b'',b'one',b'one\n\n'):
            with self.assertRaises(ValueError):m._lines(raw)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'value').write_bytes(b'valid')
            for path in ('../value','a/./value','/absolute','value:stream','a\\value','a//value'):
                with self.assertRaises(ValueError):m._bound(m.Sources(),root,dict(path=path,bytes=5,sha256=m.sha(b'valid')))


if __name__=='__main__':unittest.main(verbosity=2)
