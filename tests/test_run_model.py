"""run_model: 同じフォルダの別ファイルを import するモデル、読み込みの使い回し、ファイルが変わった後の読み直し"""
import os, sys, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predrec as P
from test_predrec import Base, snap, DEADLINE, load, dump


class TestRunModel(Base):
    def add(self, name, val):
        with open(os.path.join(self.root, 'models', 'sib_part.py'), 'w', encoding='utf-8') as f: f.write('VAL = %r\n' % val)
        with open(os.path.join(self.root, 'models', 'sib_main.py'), 'w', encoding='utf-8') as f:
            f.write(open(os.path.join(self.root, 'models', 'trial_v0.py'), encoding='utf-8').read()
                    + '\nimport sib_part\n_p = predict\ndef predict(s):\n    o = _p(s); o["rationale"] = [{"feature": str(sib_part.VAL), "direction": "up", "target": "t"}]; return o\n')
        m = load(os.path.join(self.root, 'versions', 'trial-v0.json'))
        m.update(version=name, entry='models/sib_main.py', code_files=['models/sib_main.py', 'models/sib_part.py'])
        m['code_files_hash'] = P.code_files_hash(m['code_files'], self.root)
        dump(m, os.path.join(self.root, 'versions', name + '.json'))
        self.git('add', '-A'); self.git('commit', '-qm', name)
        return P.load_manifest(self.root, name)

    def test_sibling_import_and_reload(self):
        m1 = self.add('sib-1', 'a')
        r1, _ = P.make_record(snap(), m1, '24_20260910_01', DEADLINE, 'backtest', code_root=self.root)
        self.assertEqual(r1['rationale'][0]['feature'], 'a')
        r1b, _ = P.make_record(snap(), m1, '24_20260910_02', DEADLINE, 'backtest', code_root=self.root)  # 使い回し
        self.assertEqual(r1b['rationale'][0]['feature'], 'a')
        m2 = self.add('sib-2', 'b')   # 同じ名前・同じ大きさの別ファイルを同じ秒に書き換えた新しい版(.pyc を使うと古い中身が動く)
        r2, _ = P.make_record(snap(), m2, '24_20260910_03', DEADLINE, 'backtest', code_root=self.root)
        self.assertEqual(r2['rationale'][0]['feature'], 'b')
        with self.assertRaises(P.RecordError):   # 古い版はコードが変わったので作れない
            P.make_record(snap(), m1, '24_20260910_04', DEADLINE, 'backtest', code_root=self.root)


if __name__ == '__main__':
    unittest.main()
