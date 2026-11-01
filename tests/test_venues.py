"""24場への対応(2026-10-31)の確認。版の venues で記録できる会場を決める。venues の無い古い版は大村だけ。"""
import json, os, sys, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predrec as P
from test_predrec import Base, snap, DEADLINE, load, dump


class TestVenues(Base):
    def add_version(self, name, venues):
        m = load(os.path.join(self.root, 'versions', 'trial-v0.json'))
        m['version'] = name
        if venues is not None: m['venues'] = venues
        dump(m, os.path.join(self.root, 'versions', name + '.json'))
        self.git('add', '-A'); self.git('commit', '-qm', 'add ' + name)
        return P.load_manifest(self.root, name)

    def test_old_version_is_omura_only(self):
        self.assertEqual(P.venues_of(self.man), ['24'])
        for rid in ('01_20260910_01', '02_20260627_01', '23_20260910_12'):
            with self.assertRaises(P.RecordError): self.make(rid)
        self.make('24_20260910_01')

    def test_multi_venue_version(self):
        man = self.add_version('trial-v0-24jo', ['01', '02', '24'])
        s = snap(); s['key'] = '01_20260910_03'
        rec, sn = P.make_record(s, man, '01_20260910_03', DEADLINE, 'live', code_root=self.root)
        self.assertEqual((rec['race']['jcd'], rec['race']['hd'], rec['race']['rno']), ('01', '20260910', 3))
        rp = P.save(rec, sn, self.root)
        self.assertIn('records/01/20260910/', rp)  # 記録の置き場所は OS によらず / 区切り(台帳と git の表記)
        self.assertTrue(rec['input']['path'].startswith('inputs/01/20260910/'))
        P.check_record(rec, man)
        with self.assertRaises(P.RecordError):  # 版の対象に無い会場
            P.make_record(snap(), man, '13_20260910_01', DEADLINE, 'live', code_root=self.root)

    def test_bad_race_ids(self):
        man = self.add_version('trial-v0-all', ['%02d' % i for i in range(1, 25)])
        for rid in ('00_20260910_01', '25_20260910_01', '1_20260910_01', '24_2026091_01'):
            with self.assertRaises(P.RecordError): P.make_record(snap(), man, rid, DEADLINE, 'live', code_root=self.root)

    def test_bad_venues_in_manifest(self):
        for v in (['25'], ['24', '24'], ['1'], []):
            m = load(os.path.join(self.root, 'versions', 'trial-v0.json')); m['venues'] = v
            with self.assertRaises(Exception): P.check_manifest(m)

    def test_jcd_mismatch_detected(self):
        man = self.add_version('trial-v0-2jo', ['01', '24'])
        rec, _ = P.make_record(snap(), man, '01_20260910_03', DEADLINE, 'live', code_root=self.root)
        rec['race']['jcd'] = '24'; rec['integrity']['record_hash'] = P.record_hash(rec)
        with self.assertRaises(P.RecordError): P.check_record(rec, man)

    def test_venue_removed_after_record_detected(self):
        # 版の設定を後から書き換えて会場を外す/足すと manifest_hash が変わり検出される
        man = self.add_version('trial-v0-2jo', ['01', '24'])
        rec, _ = P.make_record(snap(), man, '01_20260910_03', DEADLINE, 'live', code_root=self.root)
        man2 = dict(man); man2['venues'] = ['24']
        with self.assertRaises(P.RecordError): P.check_record(rec, man2)


if __name__ == '__main__':
    unittest.main()


class TestCoverageVenues(unittest.TestCase):
    """記録漏れの対象レースは版の対象会場だけで数える"""
    def setUp(self):
        import evaluate as E; self.E = E
        def off(j, r): return {'jcd': j, 'hd': '20260910', 'rno': r, 'course': [1, 2, 3, 4, 5, 6], 'excluded': False}
        self.res = {'%s_20260910_%02d' % (j, r): off(j, r) for j in ('01', '13', '24') for r in (1, 2, 3)}
        self.pairs = [(1, {'mode': 'backtest', 'created_at': '2026-09-10T00:00:00+00:00',
                           'race': {'race_id': '24_20260910_01', 'hd': '20260910'}}, {})]

    def test_old_version_counts_omura_only(self):
        c = self.E.coverage(None, self.pairs, {'version': 'x'}, self.res, True)
        self.assertEqual(c['n_target'], 3); self.assertEqual(c['n_missing'], 2)

    def test_multi_venue_counts_listed_venues(self):
        c = self.E.coverage(None, self.pairs, {'version': 'x', 'venues': ['01', '24']}, self.res, True)
        self.assertEqual(c['n_target'], 6); self.assertEqual(c['n_missing'], 5)


T13 = '''import json
C13 = %r
def predict(s):
    p = {c: 0.0 for c in C13}; p['3攻め成立'] = 0.4; p['2差しが制す'] = 0.35; p['その他'] = 0.25
    return {'origin': {'2まくり': 0.1, '3まくり': 0.4, '4まくり': 0.1, '5まくり': 0.1, '6まくり': 0.0, '3まくり差し': 0.2, 'その他': 0.1},
            'premise': {'entry': {'courses': [1, 2, 3, 4, 5, 6], 'p_as_predicted': None}, 'st': None, 'form': None},
            'scenario_id': None, 'boats': None, 'boat1': None, 'rationale': [], 'class13': p}
''' % (P.CLASS13,)


class TestClass13(Base):
    """13類の確率を任意で記録できる(2026-10-31 Q5)"""
    def setUp(self):
        super().setUp()
        with open(os.path.join(self.root, 'models', 't13.py'), 'w', encoding='utf-8') as f: f.write(T13)
        m = load(os.path.join(self.root, 'versions', 'trial-v0.json'))
        m.update(version='t13', entry='models/t13.py', code_files=['models/t13.py'], venues=['01', '24'])
        m['code_files_hash'] = P.code_files_hash(m['code_files'], self.root)
        dump(m, os.path.join(self.root, 'versions', 't13.json'))
        self.git('add', '-A'); self.git('commit', '-qm', 't13')
        self.m13 = P.load_manifest(self.root, 't13')

    def test_recorded_and_checked(self):
        rec, s = P.make_record(snap(), self.m13, '01_20260910_01', DEADLINE, 'live', code_root=self.root)
        self.assertEqual(rec['class13']['classes'], P.CLASS13)
        self.assertAlmostEqual(sum(rec['class13']['p']), 1.0)
        P.check_record(rec, self.m13)
        P.save(rec, s, self.root)
        self.assertEqual(P.verify(self.root, replay=True), [])

    def test_old_model_has_no_class13(self):
        rec, _ = self.make()
        self.assertNotIn('class13', rec)

    def test_bad_sum_and_order(self):
        rec, _ = P.make_record(snap(), self.m13, '24_20260910_01', DEADLINE, 'live', code_root=self.root)
        bad = json.loads(json.dumps(rec)); bad['class13']['p'][0] += 0.1; bad['integrity']['record_hash'] = P.record_hash(bad)
        with self.assertRaises(P.RecordError): P.check_record(bad, self.m13)
        bad = json.loads(json.dumps(rec)); bad['class13']['classes'] = list(reversed(P.CLASS13)); bad['integrity']['record_hash'] = P.record_hash(bad)
        with self.assertRaises(P.RecordError): P.check_record(bad, self.m13)

    def test_replay_detects_class13_edit(self):
        rec, s = P.make_record(snap(), self.m13, '24_20260910_01', DEADLINE, 'live', code_root=self.root)
        p = rec['class13']['p']; p[0], p[5] = p[5], p[0]; rec['integrity']['record_hash'] = P.record_hash(rec)
        self.assertIsNotNone(P._replay_problem(rec, s, self.m13, self.root))


class TestPrivateInputs(Base):
    """publish_inputs: false の版は入力の写しをコミットしない。写しの無い複製でも verify は通る"""
    def add(self, name, pub):
        m = load(os.path.join(self.root, 'versions', 'trial-v0.json'))
        m['version'] = name; m['venues'] = ['01', '24']
        if pub is not None: m['publish_inputs'] = pub
        dump(m, os.path.join(self.root, 'versions', name + '.json'))
        self.git('add', '-A'); self.git('commit', '-qm', name)
        return P.load_manifest(self.root, name)

    def test_private_inputs_not_committed(self):
        mp = self.add('t-priv', False); mo = self.add('t-pub', None)
        r1, s1 = P.make_record(snap(), mp, '01_20260910_01', DEADLINE, 'live', code_root=self.root); P.save(r1, s1, self.root)
        r2, s2 = P.make_record(snap(), mo, '24_20260910_02', DEADLINE, 'live', code_root=self.root); P.save(r2, s2, self.root)
        P.commit(self.root)
        tracked = self.git('ls-files').split()
        self.assertNotIn(r1['input']['path'], tracked)
        self.assertIn(r2['input']['path'], tracked)
        self.assertIn(P.paths_for('01_20260910_01', 't-priv')[0], tracked)
        self.assertEqual(P.verify(self.root, replay=True), [])  # 持ち主の手元(写しあり)
        os.remove(os.path.join(self.root, r1['input']['path']))  # 公開リポジトリを複製した人の側(写しなし)
        self.assertEqual(P.verify(self.root, replay=True), [])

    def test_private_input_pushed_is_problem(self):
        mp = self.add('t-priv', False)
        r, s = P.make_record(snap(), mp, '01_20260910_01', DEADLINE, 'live', code_root=self.root); P.save(r, s, self.root)
        P.commit(self.root)
        self.git('add', '-f', r['input']['path']); self.git('commit', '-qm', 'oops')  # 手で git add -A などをして載せてしまった
        self.assertTrue(any('入力の写しが git に載っている' in x for x in P.verify(self.root, replay=True)))

    def test_public_inputs_missing_is_problem(self):
        mo = self.add('t-pub', None)
        r, s = P.make_record(snap(), mo, '24_20260910_02', DEADLINE, 'live', code_root=self.root); P.save(r, s, self.root)
        os.remove(os.path.join(self.root, r['input']['path']))
        self.assertNotEqual(P.verify(self.root, replay=True), [])

    def test_private_input_tampered_detected(self):
        mp = self.add('t-priv', False)
        r, s = P.make_record(snap(), mp, '01_20260910_01', DEADLINE, 'live', code_root=self.root); P.save(r, s, self.root)
        p = os.path.join(self.root, r['input']['path']); d = load(p); d['before'] = {'windSpeed': 9}; dump(d, p)
        self.assertNotEqual(P.verify(self.root, replay=True), [])
