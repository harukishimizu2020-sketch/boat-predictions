"""tools/daily_live.py の確認(取得はにせものの取得モジュールで置き換える)"""
import datetime as dt, json, os, sys, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tools')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predrec as P
import daily_live as D
from _fsutil import rmtree_force
from test_predrec import Base, load, dump

UTC = dt.timezone.utc


class FakeFetcher:
    def __init__(self, hd, deadlines, fail=()):
        self.hd, self.deadlines, self.fail, self.calls = hd, deadlines, set(fail), []

    def racelist(self, hd):
        return [{'race_id': rid, 'deadline': d, 'deadline_source': 'test', 'card': {'n': 1}} for rid, d in self.deadlines.items()]

    def before(self, rid):
        self.calls.append(rid)
        if rid in self.fail: raise RuntimeError('取れない')
        return {'captured_at': (dt.datetime.now(UTC) - dt.timedelta(seconds=5)).replace(microsecond=0).isoformat(),
                'sources': ['https://example/' + rid], 'raw': {'x': 1}, 'normalized': {'windSpeed': 2}}


class TestDaily(Base):
    def setUp(self):
        super().setUp()
        m = load(os.path.join(self.root, 'versions', 'trial-v0.json'))
        m.update(version='t24', venues=['%02d' % i for i in range(1, 25)], publish_inputs=False,
                 sample_rule={'seed': 'test-seed', 'rate': 0.5, 'always': ['24']})
        dump(m, os.path.join(self.root, 'versions', 't24.json'))
        self.git('add', '-A'); self.git('commit', '-qm', 't24')
        self.state = os.path.join(self.root, '..', os.path.basename(self.root) + '_state')
        self.now = dt.datetime.now(UTC).replace(microsecond=0)
        self.hd = (self.now + P.JST).strftime('%Y%m%d')

    def tearDown(self):
        rmtree_force(self.state, ignore_errors=True)
        super().tearDown()

    def dl(self, minutes):
        return (self.now + dt.timedelta(minutes=minutes) + P.JST).replace(tzinfo=dt.timezone(P.JST)).isoformat()

    def test_card_fetched_at_tick(self):
        class LazyCard(FakeFetcher):  # 朝は締切時刻だけ、出走表は記録の直前に取る
            def racelist(self, hd):
                return [{'race_id': rid, 'deadline': d, 'deadline_source': 'test'} for rid, d in self.deadlines.items()]
            def card(self, rid):
                self.calls.append('card:' + rid); return {'n': 2, 'race': rid}
        f = LazyCard(self.hd, {'01_%s_01' % self.hd: self.dl(5), '13_%s_02' % self.hd: self.dl(60)})
        r = D.tick(self.root, self.state, f, 't24', self.hd, t=self.now)
        self.assertEqual(r['made'], ['01_%s_01' % self.hd])
        self.assertEqual([c for c in f.calls if c.startswith('card:')], ['card:01_%s_01' % self.hd])  # 締切の近いレースだけ
        inp = load(os.path.join(self.root, P.paths_for('01_%s_01' % self.hd, 't24')[1]))
        self.assertEqual(inp['card'], {'n': 2, 'race': '01_%s_01' % self.hd})

    def test_tick_refuses_without_git_identity(self):
        self.git('config', '--unset', 'user.email')
        f = FakeFetcher(self.hd, {'01_%s_01' % self.hd: self.dl(5)})
        from unittest import mock
        with mock.patch.dict(os.environ, {'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1'}):  # この PC の全体設定も見ない
            with self.assertRaises(P.RecordError):
                D.tick(self.root, self.state, f, 't24', self.hd, t=self.now)
        self.assertEqual(P._read_ledger(self.root), [])  # 記録は1件も作っていない

    def test_tick_records_due_and_misses_late(self):
        f = FakeFetcher(self.hd, {'01_%s_01' % self.hd: self.dl(5), '13_%s_02' % self.hd: self.dl(60),
                                  '24_%s_03' % self.hd: self.dl(0.5)})
        r = D.tick(self.root, self.state, f, 't24', self.hd, lead_min=8, margin_s=60, t=self.now)
        self.assertEqual(r['made'], ['01_%s_01' % self.hd])          # 締切5分前 → 記録
        self.assertEqual([m['race_id'] for m in r['missed']], ['24_%s_03' % self.hd])  # 締切30秒前(margin 60秒以内) → 漏れ
        self.assertEqual(r['waiting'], 1)                              # 60分前 → まだ
        self.assertIsNotNone(r['commit'])
        tracked = self.git('ls-files').split()
        self.assertFalse(any(x.startswith('inputs/') for x in tracked))  # 入力の写しは公開しない版
        self.assertEqual(P.verify(self.root, replay=True), [])
        r2 = D.tick(self.root, self.state, f, 't24', self.hd, t=self.now)  # 2回目は同じレースを作らない
        self.assertEqual(r2['made'], []); self.assertEqual(f.calls.count('01_%s_01' % self.hd), 1)

    def test_fetch_failure_retries_then_miss(self):
        rid = '05_%s_01' % self.hd
        f = FakeFetcher(self.hd, {rid: self.dl(5)}, fail={rid})
        r = D.tick(self.root, self.state, f, 't24', self.hd, t=self.now)
        self.assertEqual(r['made'], []); self.assertEqual(r['missed'], [])
        later = self.now + dt.timedelta(minutes=4, seconds=30)
        r = D.tick(self.root, self.state, f, 't24', self.hd, t=later)
        self.assertEqual([m['race_id'] for m in r['missed']], [rid])

    def test_plan_is_fixed(self):
        f = FakeFetcher(self.hd, {'01_%s_01' % self.hd: self.dl(30)})
        D.plan(self.state, f, self.hd)
        f.deadlines = {'01_%s_01' % self.hd: self.dl(90)}
        self.assertEqual(D.plan(self.state, f, self.hd)['races'][0]['deadline'], self.dl(30))

    def test_plan_rejects_bad_deadline(self):
        f = FakeFetcher(self.hd, {'01_%s_01' % self.hd: self.now.isoformat()})  # UTC は拒否
        with self.assertRaises(P.RecordError): D.plan(self.state, f, self.hd)

    def test_sample_rule(self):
        man = P.load_manifest(self.root, 't24')
        res = {}
        for j in ('01', '13', '24'):
            for r in range(1, 13):
                k = '%s_%s_%02d' % (j, self.hd, r)
                res[k] = {'jcd': j, 'hd': self.hd, 'rno': r, 'course': [1, 2, 3, 4, 5, 6], 'excluded': r == 12}
        s = D.sample(self.state, man, res, self.hd)
        ids = [p['race_id'] for p in s['picks']]
        self.assertEqual(sum(1 for x in ids if x.startswith('24_')), 11)  # 大村は全部(除外の12Rは外す)
        self.assertTrue(all(not x.endswith('_12') for x in ids))
        self.assertEqual(D.sample(self.state, man, res, self.hd), s)   # 同じ日の標本は動かない
        other = [x for x in ids if not x.startswith('24_')]
        self.assertTrue(0 < len(other) < 22)


if __name__ == '__main__':
    unittest.main()
