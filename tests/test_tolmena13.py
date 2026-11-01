"""本モデル tolmena13-a-p1101 の確認: 版の設定が通り、本番の形の snapshot(形は試運転 2026-11-01 尼崎8R と同じ。値は作り物で、実データは載せない)で
snapshot_builder → 記録 → verify(再実行)が通る。結果の蓄積が空でも動く。基準の版と同じ入力で並べて記録できる"""
import datetime as dt, json, os, subprocess, sys, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tools')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predrec as P
import daily_live as D
from _fsutil import rmtree_force
from test_predrec import Base, load

UTC = dt.timezone.utc
SNAP = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'snap_live_13.json'))


class Fetch:
    def __init__(self, hd, dl): self.hd, self.dl, self.calls = hd, dl, []
    def racelist(self, hd):
        return [{'race_id': '13_%s_08' % hd, 'deadline': self.dl, 'deadline_source': 'test', 'race_class': 'is-G1b'}]
    def card(self, rid): return SNAP['card']
    def before(self, rid):
        self.calls.append(rid)
        return {'captured_at': (dt.datetime.now(UTC) - dt.timedelta(seconds=5)).replace(microsecond=0).isoformat(),
                'sources': [], 'raw': {}, 'normalized': SNAP['before']}


class TestTolmena13(Base):
    def setUp(self):
        super().setUp()
        self.state = os.path.join(self.root, '..', os.path.basename(self.root) + '_state')

    def tearDown(self):
        rmtree_force(self.state, ignore_errors=True); super().tearDown()

    def test_side_by_side_with_baseline(self):
        man = P.load_manifest(self.root, 'tolmena13-a-p1101')
        P.check_manifest(man)
        now = dt.datetime.now(UTC).replace(microsecond=0)
        hd = (now + P.JST).strftime('%Y%m%d')
        dl = (now + dt.timedelta(minutes=5) + P.JST).replace(tzinfo=dt.timezone(P.JST)).isoformat()
        f = Fetch(hd, dl)
        store = self.state + '_res.jsonl'   # 結果の蓄積(1行だけ。履歴がほぼ無くても動くこと)
        prev = (dt.datetime.strptime(hd, '%Y%m%d') - dt.timedelta(days=3)).strftime('%Y%m%d')
        with open(store, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps({'key': '13_%s_01' % prev, 'jcd': '13', 'hd': prev, 'rno': 1, 'order': [1, 2, 3, 4, 5, 6], 'kimarite': '逃げ',
                                 'course': [1, 2, 3, 4, 5, 6], 'st': {str(b): '.15' for b in range(1, 7)},
                                 'tobans': {str(b): str(9000 + b) for b in range(1, 7)},
                                 'excluded': False, 'flags': []}, ensure_ascii=False) + '\n')
        try:
            r = D.tick(self.root, self.state, f, 'baseline24-v0,tolmena13-a-p1101', hd, t=now, ctx={'results_store': store})
        finally:
            os.remove(store)
        rid = '13_%s_08' % hd
        self.assertEqual(sorted(r['made']), [rid + '@baseline24-v0', rid + '@tolmena13-a-p1101'])
        self.assertEqual(len(f.calls), 1)
        rec = load(os.path.join(self.root, P.paths_for(rid, 'tolmena13-a-p1101')[0]))
        c = rec['class13']['conf']
        self.assertTrue(0 < c['sub'] <= c['main'] <= 1)
        inp = load(os.path.join(self.root, P.paths_for(rid, 'tolmena13-a-p1101')[1]))
        self.assertEqual(inp['race_class'], 'is-G1b')
        self.assertEqual(P.verify(self.root, replay=True), [])

    def test_r1102_side_by_side_with_program(self):
        """判定し直し後の2つの版(A と番組表つき)を、p1101・基準の版と同じ取得で並べて記録する。
        同じ名前のファイル(tm13_*.py)を持つ版が1つの処理で混ざらないこと: 並べて作った記録が、版ごとに別の処理で作った値と同じ"""
        vs = ['baseline24-v0', 'tolmena13-a-p1101', 'tolmena13-a-r1102', 'tolmena13-ai-r1102']
        now = dt.datetime.now(UTC).replace(microsecond=0)
        hd = (now + P.JST).strftime('%Y%m%d')
        dl = (now + dt.timedelta(minutes=5) + P.JST).replace(tzinfo=dt.timezone(P.JST)).isoformat()
        f = Fetch(hd, dl)
        store = self.state + '_res.jsonl'; pdir = self.state + '_prog'
        os.makedirs(pdir, exist_ok=True)
        with open(store, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps({'key': '13_20250101_01', 'jcd': '13', 'hd': '20250101', 'rno': 1, 'order': [1, 2, 3, 4, 5, 6], 'kimarite': '逃げ',
                                 'course': [1, 2, 3, 4, 5, 6], 'st': {str(b): '.15' for b in range(1, 7)},
                                 'tobans': {str(b): str(9000 + b) for b in range(1, 7)}, 'excluded': False, 'flags': []}, ensure_ascii=False) + '\n')
        with open(os.path.join(pdir, 'b%s.txt' % hd[2:]), 'w', encoding='cp932') as fh:
            fh.write(program_text())
        try:
            r = D.tick(self.root, self.state, f, ','.join(vs), hd, t=now, ctx={'results_store': store, 'program_dir': pdir})
        finally:
            os.remove(store)
        rid = '13_%s_08' % hd
        self.assertEqual(sorted(r['made']), sorted('%s@%s' % (rid, v) for v in vs))
        self.assertEqual(len(f.calls), 1)
        inp = {v: load(os.path.join(self.root, P.paths_for(rid, v)[1])) for v in vs}
        rec = {v: load(os.path.join(self.root, P.paths_for(rid, v)[0])) for v in vs}
        prog = inp['tolmena13-ai-r1102']['program']
        self.assertEqual(prog['day_label'], '第\u3000３日'); self.assertEqual(prog['boats']['1']['toban'], '9001')
        # r1102 の A は p1101 と木・表が同じ(判定し直しの残り147レースは学習の期間 8/31 より後だった)ので、出力も同じ
        self.assertEqual(rec['tolmena13-a-p1101']['class13'], rec['tolmena13-a-r1102']['class13'])
        self.assertNotEqual(rec['tolmena13-a-r1102']['class13'], rec['tolmena13-ai-r1102']['class13'])  # 番組表の特徴が効く
        self.assertEqual(P.verify(self.root, replay=True), [])
        # 版ごとに別の処理(読み込みの使い回しなし)で、保存した入力からもう一度出す → 並べて作った記録と同じ
        for v in vs[1:]:
            code = ('import json,sys; sys.path.insert(0, %r); import predrec as P; m = P.load_manifest(%r, %r); '
                    's = json.load(open(%r, encoding="utf-8")); '
                    'print(json.dumps(P.normalize_output(P.run_model(m, s, %r))["class13"]))') % (
                HERE, self.root, v, os.path.join(self.root, P.paths_for(rid, v)[1]), self.root)
            out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, encoding='utf-8', check=True).stdout
            self.assertEqual(json.loads(out), rec[v]['class13']['p'], v)
        rmtree_force(pdir, ignore_errors=True)


def program_text():
    """番組表 B ファイルの形(値は作り物。選手名・登番は実在しない)。尼崎(13)の8R だけ"""
    L = ['STARTB', '13BBGN', 'ボートレース尼\u3000崎   １１月\u3000２日  試験の節名            第\u3000３日', '',
         '\u3000８Ｒ  予選\u3000\u3000\u3000\u3000          Ｈ１８００ｍ  電話投票締切予定１５：００ ',
         '-' * 79, '艇 選手 選手  年 支 体級    全国      当地     モーター   ボート   今節成績  早', '-' * 79]
    ks = ['1 2 13 2  ', '3 4 2 1   ', '5 361 4   ', '2 6 45    ', '4 5 6 3   ', '6 1 5 2 F ']
    for b in range(1, 7):
        L.append('%d %d試験選手%d兵庫5%dB1 %4.2f %5.2f %4.2f %5.2f %2d %5.2f %2d %5.2f %s' % (
            b, 9000 + b, 25 + 3 * b, b % 4, 7.2 - 0.4 * b, 50 - 3 * b, 6.8 - 0.3 * b, 45 - 2 * b, 10 + b, 30 + b, 40 + b, 25 + 2 * b, ks[b - 1]))
    L += ['', '13BEND', 'FINALB', '']
    return '\n'.join(L)


if __name__ == '__main__':
    unittest.main()
