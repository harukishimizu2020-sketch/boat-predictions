"""predrec.py の確認。実行: このフォルダの親で python -m unittest discover -s tests"""
import copy, datetime as dt, json, os, shutil, subprocess, sys, tempfile, unittest
from unittest import mock
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import predrec as P

DEADLINE = '2099-01-01T10:52:00+09:00'
GIT_ENV = dict(GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')


def snap():
    cap = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)).replace(microsecond=0).isoformat()
    return {'key': '24_20260910_01', 'captured_at': cap, 'card': [{'boat': 1, 'startAvg': 0.18}], 'before': {'windSpeed': 1}}


def load(p):
    with open(p, encoding='utf-8') as f: return json.load(f)


def dump(obj, p):
    with open(p, 'w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False)


class Base(unittest.TestCase):
    """一時フォルダに版・モデル・スキーマを写し、git にコミットした状態から始める"""
    def setUp(self):
        self.root = tempfile.mkdtemp()
        for d in ('versions', 'models', 'schema'):
            shutil.copytree(os.path.join(HERE, d), os.path.join(self.root, d))
        self.git('init', '-q'); self.git('config', 'user.name', 't'); self.git('config', 'user.email', 't@t')
        self.git('add', '-A'); self.git('commit', '-qm', 'init')
        self.man = P.load_manifest(self.root, 'trial-v0')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def git(self, *a, env=None):
        e = dict(os.environ, **GIT_ENV)
        if env: e.update(env)
        return subprocess.run(['git', '-C', self.root] + list(a), check=True, capture_output=True, env=e, text=True).stdout

    def make(self, race='24_20260910_01', mode='live', deadline=DEADLINE, model_output=None, s=None):
        return P.make_record(s or snap(), self.man, race, deadline, mode, code_root=self.root, model_output=model_output)

    def save(self, race='24_20260910_01', **kw):
        rec, s = self.make(race, **kw)
        return P.save(rec, s, self.root), rec


class TestMake(Base):
    def test_derived_fields(self):
        r, _ = self.make()
        self.assertEqual(r['main_scenario']['origin'], '3まくり')
        self.assertEqual(r['main_scenario']['p'], 0.318)
        self.assertEqual(r['confidence']['mark'], '○')
        self.assertEqual(r['model']['output_source'], 'entry')
        self.assertEqual(r['model']['code_commit'], self.git('rev-parse', 'HEAD').strip())
        P.check_record(r, self.man)

    def test_reject_toda(self):
        with self.assertRaises(P.RecordError): self.make('02_20260627_01')

    def test_reject_after_deadline(self):
        with self.assertRaises(P.RecordError): self.make(deadline='2020-01-01T10:00:00+09:00')

    def test_now_cannot_be_given(self):
        # 作成時刻は PC の現在時刻だけ。現在が締切後なら拒否
        late = '2099-01-01T02:00:00+00:00'
        with mock.patch.object(P, 'now_utc', return_value=late):
            with self.assertRaises(P.RecordError): self.make()

    def test_deadline_must_be_jst(self):
        with self.assertRaises(P.RecordError): self.make(deadline='2099-01-01T01:52:00+00:00')

    def test_live_needs_captured_at(self):
        s = snap(); s['captured_at'] = None
        with self.assertRaises(P.RecordError): self.make(s=s)

    def test_backtest_without_deadline(self):
        r, _ = self.make(deadline=None, mode='backtest')
        self.assertIsNone(r['race']['deadline'])
        with self.assertRaises(P.RecordError): self.make(deadline=None, mode='live')

    def test_live_rejects_hand_made_output(self):
        mo = P.run_model(self.man, snap(), self.root)
        with self.assertRaises(P.RecordError): self.make(model_output=mo)
        r, _ = self.make(mode='backtest', model_output=mo)
        self.assertEqual(r['model']['output_source'], 'file')

    def test_live_needs_git(self):
        shutil.rmtree(os.path.join(self.root, '.git'))
        with self.assertRaises(P.RecordError): self.make()
        self.make(mode='backtest')  # 試しなら git が無くても作れる

    def test_prob_sum(self):
        mo = P.run_model(self.man, snap(), self.root); mo['origin']['3まくり'] += 0.01
        with self.assertRaises(P.RecordError): self.make(mode='backtest', model_output=mo)

    def test_code_changed(self):
        with open(os.path.join(self.root, 'models', 'trial_v0.py'), 'a') as f: f.write('\n# 変更\n')
        with self.assertRaises(P.RecordError): self.make()

    def test_crlf_same_hash(self):
        p = os.path.join(self.root, 'models', 'trial_v0.py')
        with open(p, 'rb') as f: b = f.read().replace(b'\n', b'\r\n')
        with open(p, 'wb') as f: f.write(b)
        self.make(mode='backtest')  # 改行コードだけの違いは同じハッシュ(Windows の git checkout 対策)

    def test_manifest_changed(self):
        r, _ = self.make()
        m2 = copy.deepcopy(self.man); m2['bands']['lo'] = 0.2
        with self.assertRaises(P.RecordError): P.check_record(r, m2)

    def test_manifest_checks(self):
        m = copy.deepcopy(self.man); m['bands'] = {'lo': 0.5, 'hi': 0.4}
        with self.assertRaises(P.RecordError): P.check_manifest(m)
        m = copy.deepcopy(self.man); m['entry'] = 'models/other.py'
        with self.assertRaises(P.RecordError): P.check_manifest(m)
        m = copy.deepcopy(self.man); m['origin_classes'] = list(reversed(P.ORIGIN_CLASSES))
        with self.assertRaises(P.RecordError): P.check_manifest(m)

    def test_tamper_detected(self):
        r, _ = self.make()
        r['rationale'].append({'feature': '展示タイム', 'direction': 'down', 'target': '3まくり'})
        with self.assertRaises(P.RecordError): P.check_record(r, self.man)

    def test_float_boat_numbers_rejected(self):
        r, _ = self.make()
        r['premise']['entry']['courses'] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        r['integrity']['record_hash'] = P.record_hash(r)
        with self.assertRaises(P.RecordError): P.check_record(r, self.man)

    def test_int_keys_in_input(self):
        s = snap(); s['by_boat'] = {1: 'a', 2: 'b'}
        rec, s2 = self.make(s=s); P.save(rec, s2, self.root); P.commit(self.root)
        self.assertEqual(P.verify(self.root), [])  # 整数キーでも保存前後でハッシュが同じ

    def test_mark_bands(self):
        b = {'lo': 0.30, 'hi': 0.40}
        self.assertEqual([P.mark_for(p, b) for p in (0.29, 0.30, 0.399, 0.40)], ['△', '○', '○', '◎'])

    def test_rounding(self):
        mo = P.run_model(self.man, snap(), self.root)
        mo['origin'] = {c: 1 / 7 for c in P.ORIGIN_CLASSES}
        r, _ = self.make(mode='backtest', model_output=mo)
        self.assertEqual(r['origin']['p'], [0.142857] * 7)

    def test_boats_validation(self):
        mo = P.run_model(self.man, snap(), self.root)
        mo['boats'] = [{'boat': b, 'move_p': {'差し': 0.6, 'まくり': 0.4, 'まくり差し': 0.0}, 'success_p': None} for b in range(2, 7)]
        self.make(mode='backtest', model_output=mo)
        mo['boats'][0]['move_p'] = {'差し': 0.5, 'まくり': 0.4, 'まくり差し': 0.1}
        with self.assertRaises(P.RecordError): self.make(mode='backtest', model_output=mo)  # 2号艇のまくり差しは起きない


class TestEvalHelpers(unittest.TestCase):
    def test_tail_design_example(self):
        p = [0.6, 0.1, 0.3]  # 設計書5.4の例: 差し60%・まくり10%・まくり差し30%
        self.assertAlmostEqual(P.tail(p, 1, 1.0), 0.10)
        self.assertAlmostEqual(P.tail(p, 2, 1.0), 0.40)

    def test_tail_tie_and_float_tolerance(self):
        self.assertAlmostEqual(P.tail([0.02, 0.02, 0.96], 0, 0.5), 0.02)
        self.assertAlmostEqual(P.tail([0.1 + 0.2, 0.3, 0.4], 0, 0.5), 0.3)  # 0.1+0.2 と 0.3 は同率

    def test_tail_u_needs_result(self):
        h = 'sha256:' + 'a' * 64
        r1 = {'jcd': '24', 'hd': '20260910', 'rno': 1, 'order': [1, 5, 6, 4, 2, 3], 'kimarite': '逃げ', 'entry': []}
        r2 = dict(r1, order=[3, 1, 2, 4, 5, 6])
        self.assertEqual(P.tail_u(h, r1), P.tail_u(h, r1))
        self.assertNotEqual(P.tail_u(h, r1), P.tail_u(h, r2))
        self.assertTrue(0 <= P.tail_u(h, r1) < 1)

    def test_st(self):
        self.assertAlmostEqual(P.st_z(0.15, 0.02, 0.19), 2.0)
        self.assertTrue(P.st_misread(2.0)); self.assertFalse(P.st_misread(1.95))

    def test_brier(self):
        self.assertAlmostEqual(P.brier([0.5, 0.5], [1, 0]), 0.5)


class TestSaveVerify(Base):
    def test_save_verify_commit(self):
        self.save('24_20260910_01'); self.save('24_20260910_02')
        self.assertIsNotNone(P.commit(self.root))
        self.assertEqual(P.verify(self.root), [])
        with self.assertRaises(P.RecordError): self.save('24_20260910_01')  # 上書きしない

    def test_verify_detects_edit(self):
        rp, _ = self.save()
        p = os.path.join(self.root, rp); r = load(p)
        r['origin']['p'] = [0.05, 0.5, 0.05, 0.05, 0.05, 0.25, 0.05]
        dump(r, p)
        self.assertTrue(P.verify(self.root))

    def test_verify_replay_detects_rehashed_edit(self):
        # 予測を書き換えてハッシュを計算し直しても、モデルを実行し直すと違いが出る
        _, rec = self.save()
        r = copy.deepcopy(rec); r['origin']['p'] = [0.097, 0.288, 0.088, 0.088, 0.024, 0.300, 0.115]
        snapshot = load(os.path.join(self.root, rec['input']['path']))
        self.assertIsNotNone(P._replay_problem(r, snapshot, self.man, self.root))
        self.assertIsNone(P._replay_problem(rec, snapshot, self.man, self.root))

    def test_verify_detects_removed_line(self):
        for i in (1, 2, 3): self.save('24_20260910_%02d' % i)
        lp = os.path.join(self.root, 'ledger.jsonl')
        with open(lp, encoding='utf-8') as f: ls = f.readlines()
        with open(lp, 'w', encoding='utf-8') as f: f.writelines([ls[0], ls[2]])
        pr = P.verify(self.root)
        self.assertTrue(any('連番' in x or '前の行' in x for x in pr))
        self.assertTrue(any('台帳に無い' in x for x in pr))

    def test_verify_ledger_record_mismatch(self):
        # 台帳では02レースの行なのに、中身は01レースの記録(同じレースに2つ置く=両張り)
        rp1, _ = self.save('24_20260910_01'); rp2, _ = self.save('24_20260910_02')
        shutil.copy(os.path.join(self.root, rp1), os.path.join(self.root, rp2))
        pr = P.verify(self.root)
        self.assertTrue(any('台帳と記録の中身' in x or '2つ' in x or '置き場所' in x for x in pr))

    def test_verify_input_changed(self):
        _, rec = self.save()
        ip = os.path.join(self.root, rec['input']['path']); s = load(ip); s['before']['windSpeed'] = 5
        dump(s, ip)
        self.assertTrue(any('入力' in x for x in P.verify(self.root)))

    def test_git_edit_after_commit(self):
        rp, _ = self.save(); P.commit(self.root)
        p = os.path.join(self.root, rp)
        r2 = load(p); r2['rationale'] = [{'feature': 'x', 'direction': 'up', 'target': '3まくり'}]
        r2['integrity']['record_hash'] = P.record_hash(r2)
        dump(r2, p)
        self.git('add', '-A'); self.git('commit', '-qm', 'edit')
        self.assertTrue(any('コミット後に変更' in x for x in P.verify(self.root)))

    def test_git_selective_delete_detected(self):
        # 最後の記録を、台帳の行・記録・入力ごと消してコミットし直す(都合の悪い記録を消す)
        self.save('24_20260910_01'); _, rec = self.save('24_20260910_02'); P.commit(self.root)
        lp = os.path.join(self.root, 'ledger.jsonl')
        with open(lp, encoding='utf-8') as f: ls = f.readlines()
        with open(lp, 'w', encoding='utf-8') as f: f.writelines(ls[:1])
        os.remove(os.path.join(self.root, P.paths_for('24_20260910_02', 'trial-v0')[0]))
        os.remove(os.path.join(self.root, rec['input']['path']))
        self.git('add', '-A'); self.git('commit', '-qm', 'drop')
        pr = P.verify(self.root)
        self.assertTrue(any('台帳: コミット' in x for x in pr))
        self.assertTrue(any('削除' in x for x in pr))

    def test_uncommitted_after_deadline(self):
        early = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)).replace(microsecond=0)
        dl = (early + dt.timedelta(seconds=30)).astimezone(dt.timezone(P.JST)).isoformat()
        s = snap(); s['captured_at'] = (early - dt.timedelta(minutes=1)).isoformat()
        with mock.patch.object(P, 'now_utc', return_value=early.isoformat()), \
             mock.patch.object(P.dt, 'datetime', wraps=dt.datetime) as m:
            m.now.return_value = early
            rec, s2 = self.make(deadline=dl, s=s)
        P.save(rec, s2, self.root)
        self.assertTrue(any('コミットされていない' in x for x in P.verify(self.root)))

    def test_git_commit_after_deadline(self):
        self.save(); self.git('add', '-A')
        self.git('commit', '-qm', 'late', env={'GIT_COMMITTER_DATE': '2099-01-02T00:00:00+09:00'})
        self.assertTrue(any('締切後' in x for x in P.verify(self.root)))


if __name__ == '__main__':
    unittest.main()
