"""24場の基準モデル baseline24-v0 の確認: 版の設定が通り、24場すべてで記録が作れ、自信度が入り、同じ入力で同じ予測になる"""
import os, sys, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predrec as P
from test_predrec import Base, snap, DEADLINE


class TestBaseline24(Base):
    def test_all_venues(self):
        man = P.load_manifest(self.root, 'baseline24-v0')
        self.assertEqual(P.venues_of(man), ['%02d' % i for i in range(1, 25)])
        self.assertFalse(P.publishes_inputs(man))
        seen = set()
        for j in range(1, 25):
            rid = '%02d_20261102_01' % j
            s = dict(snap(), key=rid)
            rec, sn = P.make_record(s, man, rid, DEADLINE, 'live', code_root=self.root)
            self.assertEqual(rec['class13']['conf']['def'], 'conf-v0')
            self.assertEqual(rec['boats'][0]['move_p']['まくり差し'], 0.0)
            seen.add(tuple(rec['origin']['p']))
            P.save(rec, sn, self.root)
        self.assertGreater(len(seen), 20)   # 会場ごとに違う確率
        P.commit(self.root)
        self.assertEqual(P.verify(self.root, replay=True), [])


if __name__ == '__main__':
    unittest.main()
