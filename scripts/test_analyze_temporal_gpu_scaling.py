import tempfile, json, unittest
from pathlib import Path
from analyze_temporal_gpu_scaling import analyze

class TemporalScalingTests(unittest.TestCase):
    def test_sorts_and_checks_counts(self):
        with tempfile.TemporaryDirectory() as td:
            ps=[]
            for n, status, eff, fail in [(40,'engineering_smoke_only',1,0),(64,'engineering_smoke_failure',0,1)]:
                p=Path(td)/f'{n}.json'; p.write_text(json.dumps({'num_frames':n,'status':status,'n_requested':1,'n_effective':eff,'n_failures':fail,'peak_cuda_memory_bytes':None})); ps.append(p)
            self.assertEqual([x['num_frames'] for x in analyze(ps)['conditions']], [40,64])
    def test_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.json'; p.write_text(json.dumps({'num_frames':40,'status':'engineering_smoke_only','n_requested':1,'n_effective':0,'n_failures':0,'peak_cuda_memory_bytes':1}))
            with self.assertRaises(ValueError): analyze([p])
