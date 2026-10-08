"""S4 本モデル(models/s4_v1.py・tools/build_s4.py・tools/backtest_s4.py)の確認。
実行: このフォルダの親で python -m unittest discover -s tests
確かめること: 予測の形・確率の合計が1・2号艇のまくり差しが0・同じ入力なら同じ出力・欠損だらけの入力でも動く・
λ=0 なら基準そのもの・履歴の特徴がその日より前の判定済みレースだけから作られる(未来を見ない)・記録の作成と verify が通る。"""
import copy, importlib.util, json, os, shutil, subprocess, sys, tempfile, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tools'))
import predrec as P
import build_s4 as S

OC = P.ORIGIN_CLASSES; MOVES = P.MOVES
GIT_ENV = dict(GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')


def load_model():
    spec = importlib.util.spec_from_file_location('s4_v1_test', os.path.join(HERE, 'models', 's4_v1.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def snapshot_full():
    """実データに似せた作り物のスナップショット(大村 2026-09-10 1R の形)"""
    card = [{'boat': b, 'name': 'x', 'grade': ['A1', 'A2', 'B1', 'B1', 'B2', 'A2'][b - 1], 'F': [0, 1, 0, 0, 0, 2][b - 1], 'L': 0,
             'startAvg': [0.14, 0.17, 0.16, 0.18, 0.15, 0.19][b - 1], 'winAll': [6.5, 5.2, 4.8, 4.1, 5.9, 3.2][b - 1],
             'win2All': [45.0, 35.0, 30.0, 25.0, 38.0, 15.0][b - 1], 'win3All': 60.0, 'motor2': [40.0, 33.0, 30.0, 28.0, 35.0, 25.0][b - 1], 'motor3': 50.0} for b in range(1, 7)]
    racers = [{'boatNumber': b, 'startSinnyu': b, 'startTenjiTime': [0.12, 0.15, 0.08, 0.20, 0.10, 0.17][b - 1], 'startTenjiRank': b,
               'tenjiRank': b, 'tenjiTime': [6.70, 6.75, 6.68, 6.80, 6.72, 6.90][b - 1], 'tilt': -0.5, 'weight': 52.0 + b} for b in range(1, 7)]
    ot = [{'boatNumber': b, 'isshuTime': 37.0 + 0.1 * b, 'mawariashiTime': 6.3 + 0.02 * b, 'chokusenTime': 7.4} for b in range(1, 7)]
    hist = {str(b): {'same_course': {'n': 8.0, '差し': 3.0, 'まくり': 2.5, 'まくり差し': 2.5},
                     'any_course': {'n': 30.0, '差し': 14.0, 'まくり': 6.0, 'まくり差し': 10.0}, 'st': {'n': 20, 'mean': 0.15 + 0.005 * b}} for b in range(1, 7)}
    return {'race_id': '24_20260910_01', 'jcd': '24', 'hd': '20260910', 'rno': 1, 'captured_at': None,
            'tobans': {str(b): '40%02d' % b for b in range(1, 7)}, 'card': card,
            'before': {'weather': '晴', 'windSpeed': 3, 'windDirection': 7, 'waveHeight': 2, 'weatherDegree': 28, 'waterDegree': 26, 'racers': racers},
            'original_tenjis': ot, 'race_class': 'is-ippan', 'title': 'test', 'history': hist, 'history_cutoff': '20260910'}


def check_output(tc, out):
    tc.assertEqual(set(out['origin']), set(OC))
    tc.assertAlmostEqual(sum(out['origin'].values()), 1.0, places=9)
    tc.assertTrue(all(0 <= v <= 1 for v in out['origin'].values()))
    tc.assertEqual(out['premise']['entry']['courses'], [1, 2, 3, 4, 5, 6])
    tc.assertEqual([s['boat'] for s in out['premise']['st']], [1, 2, 3, 4, 5, 6])
    tc.assertTrue(all(s['sd'] > 0 and 0 < s['mean'] < 0.4 for s in out['premise']['st']))
    tc.assertEqual([b['boat'] for b in out['boats']], [2, 3, 4, 5, 6])
    for b in out['boats']:
        tc.assertAlmostEqual(sum(b['move_p'].values()), 1.0, places=9)
        tc.assertTrue(all(0 <= v <= 1 for v in b['move_p'].values()))
        if b['boat'] == 2: tc.assertEqual(b['move_p']['まくり差し'], 0.0)
    tc.assertIsNone(out['boat1'])
    for r in out['rationale']:
        tc.assertEqual(set(r), {'boat', 'feature', 'value', 'target', 'direction', 'effect'})
        tc.assertIn(r['direction'], ('up', 'down'))
    P.normalize_output(out)  # 記録の形にできること


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.m = load_model(); self.params = self.m.load_params()

    def test_shape_and_sums(self):
        check_output(self, self.m.predict(snapshot_full()))

    def test_deterministic(self):
        a = self.m.predict(snapshot_full()); b = self.m.predict(json.loads(json.dumps(snapshot_full())))
        self.assertEqual(P.canon(a), P.canon(b))
        s = snapshot_full(); s['before']['racers'] = list(reversed(s['before']['racers'])); s['card'] = list(reversed(s['card']))  # 並び順は結果に関係ない
        self.assertEqual(P.canon(self.m.predict(s)), P.canon(a))

    def test_missing_everything(self):
        s = {'race_id': '24_20260910_01', 'jcd': '24', 'hd': '20260910', 'rno': 1, 'tobans': {}, 'card': None, 'before': None,
             'original_tenjis': None, 'race_class': None, 'history': None}
        out = self.m.predict(s); check_output(self, out)
        for st in out['premise']['st']:  # 過去走が無い艇はコースの切片と素の sd
            self.assertEqual(st['mean'], self.params['st']['alpha'][str(st['boat'])]); self.assertEqual(st['sd'], self.params['st']['sd0'][str(st['boat'])])
        check_output(self, self.m.predict({}))

    def test_features_match_params(self):
        names, vals = self.m.features(snapshot_full(), self.params['feature_params'])
        self.assertEqual(names, self.params['feature_names']); self.assertEqual(len(vals), len(names))
        self.assertEqual(len(self.params['scale_mean']), len(names)); self.assertEqual(len(self.params['origin_model']['coef'][0]), len(names))

    def test_features_change_prediction(self):
        """入力が変わると予測も変わる(定数モデルではない)"""
        s = snapshot_full(); a = self.m.predict(s)
        s['history']['3']['same_course'] = {'n': 20.0, '差し': 0.0, 'まくり': 20.0, 'まくり差し': 0.0}; s['before']['racers'][2]['startTenjiTime'] = -0.02
        b = self.m.predict(s)
        self.assertNotEqual(P.canon(a['origin']), P.canon(b['origin']))

    def test_lambda_zero_is_baseline(self):
        p = copy.deepcopy(self.params); p['choice'] = dict(p['choice'], candidate='base', origin_lambda=0.0, move_lambda=0.0)
        r = self.m.compute(snapshot_full(), p)
        for c, v in zip(OC, r['origin']): self.assertAlmostEqual(v, p['base_origin'][c], places=9)
        for b in range(2, 7):
            for mv, v in zip(MOVES, r['moves'][b]): self.assertAlmostEqual(v, p['base_move'][str(b)][mv], places=9)

    def test_origin_soft_same_as_evaluate(self):
        import evaluate as E
        Pm = {2: [0.9, 0.1, 0.0], 3: [0.3, 0.3, 0.4], 4: [0.5, 0.1, 0.4], 5: [0.2, 0.2, 0.6], 6: [0.7, 0.05, 0.25]}
        for a, b in zip(self.m.origin_soft(Pm), E.origin_soft(Pm)): self.assertAlmostEqual(a, b, places=12)

    def test_st_uses_history(self):
        s = snapshot_full(); s['history']['3']['st'] = {'n': 20, 'mean': 0.10}
        st = next(x for x in self.m.predict(s)['premise']['st'] if x['boat'] == 3)
        p = self.params['st']; self.assertAlmostEqual(st['mean'], p['alpha']['3'] + p['beta'] * (0.10 - p['center']), places=9)
        s['history']['3']['st'] = {'n': 2, 'mean': 0.10}  # 5走未満は使わない
        st = next(x for x in self.m.predict(s)['premise']['st'] if x['boat'] == 3)
        self.assertEqual(st['mean'], p['alpha']['3'])


class TestHistoryNoFuture(unittest.TestCase):
    """履歴の特徴は、その日(hd)より前の判定済みレース・公式結果だけから作る。同じ日と未来のレースは入らない"""
    def make_data(self):
        d = S.Data.__new__(S.Data)
        d.move_hist = {'4001': [('20260901', 3, [0.0, 1.0, 0.0]), ('20260905', 3, [0.0, 0.0, 1.0]), ('20260905', 4, [1.0, 0.0, 0.0]),
                                ('20260910', 3, [0.0, 1.0, 0.0]), ('20260920', 3, [0.0, 1.0, 0.0])]}
        d.move_dates = {t: [x[0] for x in v] for t, v in d.move_hist.items()}
        d.st_hist = {'4001': [('20260901', 1, 0.10), ('20260905', 2, 0.20), ('20260910', 3, 0.30), ('20260920', 4, 0.40)]}
        d.st_dates = {t: [x[0] for x in v] for t, v in d.st_hist.items()}
        return d

    def test_strictly_before(self):
        d = self.make_data()
        h = d.history('4001', 3, '20260910')
        self.assertEqual(h['same_course'], {'n': 2.0, '差し': 0.0, 'まくり': 1.0, 'まくり差し': 1.0})   # 09-01 と 09-05(コース3)だけ
        self.assertEqual(h['any_course'], {'n': 3.0, '差し': 1.0, 'まくり': 1.0, 'まくり差し': 1.0})    # 09-05 のコース4も
        self.assertEqual(h['st'], {'n': 2, 'mean': 0.15})                                              # 09-01 と 09-05 のST
        h2 = d.history('4001', 3, '20260921')
        self.assertEqual(h2['same_course']['n'], 4.0); self.assertEqual(h2['st']['n'], 4)
        h0 = d.history('4001', 3, '20260901')
        self.assertEqual(h0['same_course']['n'], 0.0); self.assertEqual(h0['st'], {'n': 0, 'mean': None})
        self.assertEqual(d.history(None, 3, '20260910')['same_course']['n'], 0.0)

    def test_future_does_not_change(self):
        d = self.make_data(); before = json.dumps(d.history('4001', 3, '20260910'), sort_keys=True)
        d.move_hist['4001'].append(('20261001', 3, [0.0, 1.0, 0.0])); d.move_dates['4001'].append('20261001')
        d.st_hist['4001'].append(('20261001', 5, 0.01)); d.st_dates['4001'].append('20261001')
        self.assertEqual(json.dumps(d.history('4001', 3, '20260910'), sort_keys=True), before)

    def test_st_last_20(self):
        d = self.make_data()
        d.st_hist['4001'] = [('2026%02d%02d' % (1 + i // 28, 1 + i % 28), 1, 0.1 + 0.01 * i) for i in range(30)]
        d.st_dates['4001'] = [x[0] for x in d.st_hist['4001']]
        h = d.history('4001', 1, '20270101')['st']
        self.assertEqual(h['n'], 20); self.assertAlmostEqual(h['mean'], sum(0.1 + 0.01 * i for i in range(10, 30)) / 20, places=6)

    def test_snapshot_has_no_odds_or_actual_st(self):
        d = self.make_data()
        d.res = {'24_20260910_01': {'jcd': '24', 'hd': '20260910', 'rno': 1, 'order': [1, 2, 3, 4, 5, 6], 'kimarite': '逃げ', 'course': [1, 2, 3, 4, 5, 6],
                                    'excluded': False, 'flags': [], 'tobans': {str(b): '4001' if b == 3 else None for b in range(1, 7)},
                                    'entry': [{'boat': b, 'st': '.15'} for b in range(1, 7)]}}
        d.aux = {'24_20260910_01': {'key': '24_20260910_01', 'card': [{'boat': 3, 'startAvg': 0.15}], 'before': {'windSpeed': 1, 'racers': []},
                                    'originalTenjis': [], 'raceClass': 'is-ippan', 'odds3t': {'_3t123': 5.0}, 'title': 't'}}
        s = d.snapshot('24_20260910_01')
        self.assertNotIn('odds3t', json.dumps(s)); self.assertNotIn('order', s); self.assertNotIn('entry', s)
        self.assertEqual(s['history_cutoff'], '20260910'); self.assertEqual(s['history']['3']['same_course']['n'], 2.0)
        self.assertEqual(s['tobans']['3'], '4001')
        check_output(self, load_model().predict(s))


class TestRecordRoundtrip(unittest.TestCase):
    """版 s4-v1 で記録を作り、verify(モデルの再実行で一致)が通ること"""
    def setUp(self):
        self.root = tempfile.mkdtemp()
        for d in ('versions', 'models', 'schema'):
            shutil.copytree(os.path.join(HERE, d), os.path.join(self.root, d), ignore=shutil.ignore_patterns('__pycache__'))
        e = dict(os.environ, **GIT_ENV)
        for a in (['init', '-q'], ['config', 'user.name', 't'], ['config', 'user.email', 't@t'], ['add', '-A'], ['commit', '-qm', 'init']):
            subprocess.run(['git', '-C', self.root] + a, check=True, capture_output=True, env=e)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_make_and_verify(self):
        man = P.load_manifest(self.root, 's4-v1')
        self.assertEqual(man['soft_label_table'], 'labels/soft-v2.json')
        self.assertEqual(man['code_files_hash'], P.code_files_hash(man['code_files'], HERE))
        rec, snap = P.make_record(snapshot_full(), man, '24_20260910_01', None, 'backtest', code_root=self.root)
        P.save(rec, snap, self.root)
        self.assertEqual(rec['confidence']['mark'], P.mark_for(rec['main_scenario']['p'], man['bands']))
        self.assertEqual(P.verify(self.root, replay=True), [])
        # 記録の予測を書き換えると再実行で検出される
        rp = os.path.join(self.root, P.paths_for('24_20260910_01', 's4-v1')[0])
        with open(rp, encoding='utf-8') as f: r = json.load(f)
        r['origin']['p'][1], r['origin']['p'][5] = r['origin']['p'][5], r['origin']['p'][1]
        o, pm = P.main_scenario(r['origin']['p']); r['main_scenario'] = {'origin': o, 'p': pm, 'scenario_id': None}
        r['confidence']['mark'] = P.mark_for(pm, man['bands']); r['integrity']['record_hash'] = P.record_hash(r)
        with open(rp, 'w', encoding='utf-8') as f: json.dump(r, f, ensure_ascii=False)
        self.assertTrue(any('違う' in p for p in P.verify(self.root, replay=True)))


if __name__ == '__main__':
    unittest.main()
