"""13類の自信度(conf-v0)の記録と較正表の確認(2026-10-31)。
主 = その他を除く上位3つの確率の合計、副 = その他を除く最大確率。値は predrec が class13 から計算し、手で書いた値は check_record が拒む。"""
import json, os, random, sys, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predrec as P
import evaluate as E
from test_predrec import Base, snap, DEADLINE, load, dump
from test_venues import T13


class TestConf(Base):
    def setUp(self):
        super().setUp()
        with open(os.path.join(self.root, 'models', 't13.py'), 'w', encoding='utf-8') as f: f.write(T13)
        for name, cd in (('tc', 'conf-v0'), ('tn', None)):
            m = load(os.path.join(self.root, 'versions', 'trial-v0.json'))
            m.update(version=name, entry='models/t13.py', code_files=['models/t13.py'], venues=['01', '24'])
            if cd: m['confidence_def'] = cd
            m['code_files_hash'] = P.code_files_hash(m['code_files'], self.root)
            dump(m, os.path.join(self.root, 'versions', name + '.json'))
        self.git('add', '-A'); self.git('commit', '-qm', 'tc')
        self.mc, self.mn = P.load_manifest(self.root, 'tc'), P.load_manifest(self.root, 'tn')

    def test_conf_v0_values(self):
        rec, s = P.make_record(snap(), self.mc, '01_20260910_01', DEADLINE, 'live', code_root=self.root)
        c = rec['class13']['conf']
        # 3攻め成立 0.4・2差しが制す 0.35・その他 0.25(その他は数えない)。3つ目は 0 の同率で CLASS13 の並び順の先 = 2攻め・1残す
        self.assertEqual(c, {'def': 'conf-v0', 'main': 0.75, 'sub': 0.4, 'top3': ['3攻め成立', '2差しが制す', '2攻め・1残す']})
        P.save(rec, s, self.root)
        self.assertEqual(P.verify(self.root, replay=True), [])

    def test_no_def_no_conf(self):
        rec, _ = P.make_record(snap(), self.mn, '01_20260910_01', DEADLINE, 'live', code_root=self.root)
        self.assertNotIn('conf', rec['class13'])

    def test_hand_written_conf_rejected(self):
        rec, _ = P.make_record(snap(), self.mc, '01_20260910_01', DEADLINE, 'live', code_root=self.root)
        bad = json.loads(json.dumps(rec)); bad['class13']['conf']['main'] = 0.9; bad['integrity']['record_hash'] = P.record_hash(bad)
        with self.assertRaises(P.RecordError): P.check_record(bad, self.mc)
        bad = json.loads(json.dumps(rec)); del bad['class13']['conf']; bad['integrity']['record_hash'] = P.record_hash(bad)
        with self.assertRaises(P.RecordError): P.check_record(bad, self.mc)
        with self.assertRaises(P.RecordError): P.check_record(rec, self.mn)  # 定義の無い版に自信度

    def test_def_needs_class13_and_known_name(self):
        m = dict(self.man, confidence_def='conf-v0')
        with self.assertRaises(P.RecordError):
            P.make_record(snap(), m, '24_20260910_01', DEADLINE, 'backtest', code_root=self.root)  # 13類を出さないモデル
        with self.assertRaises(P.RecordError): P.check_manifest(dict(self.man, confidence_def='conf-zzz'))

    def test_conf_report(self):
        rids = ['01_20260910_%02d' % i for i in range(1, 4)]
        for rid in rids:
            rec, s = P.make_record(snap(), self.mc, rid, DEADLINE, 'live', code_root=self.root); P.save(rec, s, self.root)
        tp = os.path.join(self.root, '..', os.path.basename(self.root) + '_truth.jsonl')
        with open(tp, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'race_id': rids[0], 'p': {'3攻め成立': 1.0}}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'race_id': rids[1], 'p': {'2攻め・1残す': 0.5, 'その他': 0.5}}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'race_id': rids[2], 'p': {'5が制す': 1.0}}, ensure_ascii=False) + '\n')
        try:
            r = E.conf_report(self.root, 'tc', tp)
        finally:
            os.remove(tp)
        a = r['all']
        self.assertEqual(r['defs'], ['conf-v0'])
        self.assertEqual(a['main']['n'], 3)
        self.assertAlmostEqual(a['main']['mean_hit'], (1.0 + 0.5 + 0.0) / 3)
        self.assertAlmostEqual(a['sub']['mean_hit'], 1.0 / 3)
        self.assertEqual(list(r['by_venue']), ['01'])


class TestCalibration(unittest.TestCase):
    def test_bands_merge_and_ci(self):
        rnd = random.Random(1)
        conf = [rnd.uniform(0.4, 0.8) for _ in range(400)] + [0.95] * 10   # 0.9〜1.0 は10件だけ → 前の帯にまとめる
        hit = [min(1.0, max(0.0, c + rnd.gauss(0, 0.2))) for c in conf]
        r = E.calibration(conf, hit)
        self.assertEqual(sum(b['n'] for b in r['bands']), 410)
        self.assertTrue(all(b['n'] >= 30 for b in r['bands']))
        self.assertEqual(r['bands'][-1]['hi'], 1.0)
        for b in r['bands']: self.assertLessEqual(b['ci_lo'], b['hit']); self.assertLessEqual(b['hit'], b['ci_hi'])
        self.assertLess(r['ece'], 0.05)

    def test_edges_match_numpy_digitize(self):
        # 帯の境は「lo ≤ x < hi」。0.1 はちょうど 0.1〜0.2 の帯、1.0 は最後の帯
        r = E.calibration([0.1] * 30 + [1.0] * 30, [0.0] * 60)
        self.assertEqual([(b['lo'], b['hi']) for b in r['bands']], [(0.0, 0.2), (0.2, 1.0)])

    def test_bad_truth(self):
        rec = {'race': {'race_id': '01_20260910_01', 'jcd': '01'}, 'class13': {'conf': {'def': 'conf-v0', 'main': .5, 'sub': .3, 'top3': ['a', 'b', 'c']}}}
        with self.assertRaises(E.EvalError): E.conf_hits(rec, {'無い分類': 1.0})
        with self.assertRaises(E.EvalError): E.conf_hits(rec, {'3攻め成立': 0.0})


if __name__ == '__main__':
    unittest.main()
