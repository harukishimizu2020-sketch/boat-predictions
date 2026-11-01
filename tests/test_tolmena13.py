"""本モデル tolmena13-a-p1101 の確認: 版の設定が通り、本番の形の snapshot(形は試運転 2026-11-01 尼崎8R と同じ。値は作り物で、実データは載せない)で
snapshot_builder → 記録 → verify(再実行)が通る。結果の蓄積が空でも動く。基準の版と同じ入力で並べて記録できる"""
import datetime as dt, json, os, sys, unittest
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


if __name__ == '__main__':
    unittest.main()
