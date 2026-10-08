"""evaluate.py(S3 の評価器)の確認。答えが分かっている模擬データで、採点・想定外・BSS・外れ分類・判定が設計どおりに動くかを見る。
実行: このフォルダの親で python -m unittest discover -s tests"""
import collections, hashlib, json, math, os, random, shutil, subprocess, sys, tempfile, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import predrec as P
import evaluate as E

OC = P.ORIGIN_CLASSES
# 設計書5.5 / a5_common.py の SOFT と同じ値(テスト用。本番の表は labels/ の版付きファイル)
LAB = {'version': 'test', 'moves': P.MOVES, 'high_self': 0.97,
       'table': {'差し|中': {'差し': .92, 'まくり差し': .08}, '差し|低': {'差し': .84, 'まくり差し': .16},
                 'まくり差し|中': {'まくり差し': .89, '差し': .08, 'まくり': .03}, 'まくり差し|低': {'まくり差し': .72, '差し': .20, 'まくり': .08},
                 'まくり|中': {'まくり': .62, 'まくり差し': .33, '差し': .05}, 'まくり|低': {'まくり': .32, 'まくり差し': .55, '差し': .13}},
       'fallback': {str(b): [0.6, 0.2, 0.2] for b in range(2, 7)}, 'rules': {}}
BASE = [0.097, 0.318, 0.088, 0.088, 0.024, 0.270, 0.115]


class mock_argv:
    def __init__(self, a): self.a = a
    def __enter__(self): self.old = sys.argv; sys.argv = self.a
    def __exit__(self, *e): sys.argv = self.old


def rh(i):
    return 'sha256:' + hashlib.sha256(str(i).encode()).hexdigest()


def rnd6(p):
    """記録と同じく小数6桁に丸め、合計を1にそろえる(最大の項で調整)"""
    q = [round(x, 6) for x in p]; i = max(range(len(q)), key=lambda k: q[k]); q[i] = round(q[i] + 1 - sum(q), 6)
    return q


def rec(i, p, mark='○', st=None, boats=None, courses=(1, 2, 3, 4, 5, 6), hd='20260910', mode='live', created='2026-09-10T01:00:00+00:00'):
    o, pm = P.main_scenario(p)
    return {'race': {'race_id': '24_%s_%02d' % (hd, i % 12 + 1), 'hd': hd}, 'mode': mode, 'created_at': created,
            'model': {'version': 'sim'}, 'origin': {'classes': OC, 'p': p}, 'main_scenario': {'origin': o, 'p': pm},
            'confidence': {'mark': mark}, 'premise': {'entry': {'courses': list(courses)}, 'st': st},
            'boats': boats, 'integrity': {'record_hash': rh(i)}}


def official(i, course=(1, 2, 3, 4, 5, 6), excluded=False, st=None):
    st = st or {b: 0.15 for b in range(1, 7)}
    return {'jcd': '24', 'hd': '20260910', 'rno': i % 12 + 1, 'order': [1, 2, 3, 4, 5, 6], 'kimarite': '逃げ',
            'course': list(course), 'excluded': excluded, 'entry': [{'boat': b, 'st': st[b]} for b in range(1, 7)], 'sim': i}


def judge_for(origin, conf='高'):
    """判定ラベルから作った起点(ハード)が origin になる判定結果"""
    j = {str(b): ['差し', conf] for b in range(1, 7)}
    if origin == 'その他': return j
    b = int(origin[0]); j[str(b)] = ['まくり' if origin.endswith('まくり') else 'まくり差し', conf]
    return j


def draw(p, rng):
    x = rng.random(); s = 0.0
    for i, v in enumerate(p):
        s += v
        if x < s: return i
    return len(p) - 1


def dirichlet(rng, k, a=1.0):
    g = [rng.gammavariate(a, 1) for _ in range(k)]; s = sum(g); return [x / s for x in g]


class TestLabels(unittest.TestCase):
    def test_soft_move_table(self):
        self.assertEqual(E.soft_move(3, 'まくり', '低', LAB), [.13, .32, .55])
        self.assertEqual(E.soft_move(3, 'まくり', '高', LAB), [.015, .97, .015])
        self.assertIsNone(E.soft_move(3, '判断不可', '中', LAB))
        self.assertIsNone(E.soft_move(3, None, None, LAB))

    def test_boat2_rule(self):
        lab = dict(LAB, rules={'boat2_makurizashi_to_sashi': True}, table2={'差し|中': {'差し': 1.0}, '差し|低': {'差し': .87, 'まくり': .13}})
        self.assertEqual(E.soft_move(2, 'まくり', '中', lab), [.05 + .33, .62, 0.0])
        self.assertEqual(E.soft_move(2, '差し', '低', lab), [.87, .13, 0.0])
        self.assertEqual(E.soft_move(2, 'まくり', '高', lab), [.015 + .015, .97, 0.0])
        self.assertEqual(E.soft_move(3, 'まくり', '中', lab), [.05, .62, .33])  # 3号艇には当てない

    def test_winner_rule(self):
        lab = dict(LAB, rules={'winner_kimarite_certain': True})
        off = {'order': [4, 1, 2, 3, 5, 6], 'kimarite': 'まくり'}
        self.assertEqual(E.soft_move(4, 'まくり', '低', lab, off), [0.0, 1.0, 0.0])
        self.assertEqual(E.soft_move(4, '差し', '低', lab, off), [.84, 0.0, .16])  # 決まり手と違うラベルは表どおり
        self.assertEqual(E.soft_move(3, 'まくり', '低', lab, off), [.13, .32, .55])

    def test_label_file_soft_v1(self):
        # 同梱の表(labels/soft-v1.json)が a5_common.py の soft(t, c, b, r)(2026-10-07 修正後)と同じ値を返す代表例
        lab = E.load_labels(os.path.join(HERE, 'labels', 'soft-v1.json'))
        self.assertEqual(E.soft_move(2, '差し', '中', lab), [1.0, 0.0, 0.0])
        self.assertEqual(E.soft_move(2, '差し', '低', lab), [.87, .13, 0.0])
        self.assertEqual(E.soft_move(2, 'まくり', '低', lab), [.13 + .55, .32, 0.0])
        self.assertEqual(E.soft_move(3, 'まくり', '低', lab), [.13, .32, .55])
        self.assertEqual(E.soft_move(5, 'まくり差し', '高', lab), [.015, .015, .97])
        self.assertEqual(E.soft_move(4, 'まくり', '低', lab, {'order': [4, 1, 2, 3, 5, 6], 'kimarite': 'まくり'}), [0.0, 1.0, 0.0])
        for b in range(2, 7): self.assertAlmostEqual(sum(lab['fallback'][str(b)]), 1.0, places=9)

    def test_label_file_soft_v2(self):
        # soft-v2(2026-10-08、手動判定40レース後の大村専用の表): 確信度「高」も表の値、2号艇は table2 の差し・まくり
        lab = E.load_labels(os.path.join(HERE, 'labels', 'soft-v2.json'))
        self.assertEqual(E.soft_move(3, 'まくり', '低', lab), [.1062, .3459, .5479])
        self.assertEqual(E.soft_move(5, 'まくり', '高', lab), [.003, .9296, .0674])  # high_self(0.97)ではなく表
        self.assertEqual(E.soft_move(2, 'まくり', '低', lab), [.375, .625, 0.0])
        self.assertEqual(E.soft_move(2, '差し', '高', lab), [.9943, .0057, 0.0])
        self.assertEqual(E.soft_move(2, 'まくり差し', '中', lab), [.111 + .8821, .0069, 0.0])  # table2 に無い → table から差しへ移す
        self.assertEqual(E.soft_move(4, 'まくり', '低', lab, {'order': [4, 1, 2, 3, 5, 6], 'kimarite': 'まくり'}), [0.0, 1.0, 0.0])
        for k, d in list(lab['table'].items()) + list(lab['table2'].items()): self.assertAlmostEqual(sum(d.values()), 1.0, places=9, msg=k)
        for b in range(2, 7): self.assertAlmostEqual(sum(lab['fallback'][str(b)]), 1.0, places=9)

    def test_baseline_v2_model(self):
        man = P.load_manifest(HERE, 'baseline-v2')
        self.assertEqual(man['soft_label_table'], 'labels/soft-v2.json')
        self.assertEqual(P.code_files_hash(man['code_files'], HERE), man['code_files_hash'])
        out = P.normalize_output(P.run_model(man, {}, HERE))
        self.assertAlmostEqual(sum(out['origin']), 1.0, places=6)

    def test_baseline_v1_model(self):
        man = P.load_manifest(HERE, 'baseline-v1')
        self.assertEqual(P.code_files_hash(man['code_files'], HERE), man['code_files_hash'])
        out = P.normalize_output(P.run_model(man, {}, HERE))
        self.assertAlmostEqual(sum(out['origin']), 1.0, places=6)
        self.assertEqual(len(out['premise']['st']), 6); self.assertEqual(out['boats'][0]['move_p']['まくり差し'], 0.0)

    def test_origin_hard(self):
        H = {2: '差し', 3: 'まくり差し', 4: 'まくり', 5: 'まくり', 6: '差し'}
        self.assertEqual(E.origin_hard(H), '4まくり')
        self.assertEqual(E.origin_hard({2: '差し', 3: 'まくり差し', 4: '差し', 5: '差し', 6: '差し'}), '3まくり差し')
        self.assertEqual(E.origin_hard({2: '差し', 3: '差し', 4: 'まくり差し', 5: '差し', 6: '差し'}), 'その他')  # 4まくり差しは「その他」
        self.assertEqual(E.origin_hard({b: '差し' for b in range(2, 7)}), 'その他')

    def test_origin_soft(self):
        P_ = {b: [1.0, 0.0, 0.0] for b in range(2, 7)}; P_[3] = [0.0, 1.0, 0.0]
        q = E.origin_soft(P_)
        self.assertAlmostEqual(q[OC.index('3まくり')], 1.0)
        P_ = {b: E.soft_move(b, '差し', '中', LAB) for b in range(2, 7)}; P_[3] = E.soft_move(3, 'まくり', '低', LAB)
        q = E.origin_soft(P_)
        self.assertAlmostEqual(sum(q), 1.0)
        self.assertAlmostEqual(q[OC.index('3まくり')], 0.32)
        # 2〜6号艇がすべてまくりでない確率 × 3号艇がまくり差し(まくりでない条件付き)
        self.assertAlmostEqual(q[OC.index('3まくり差し')], (1 - .32) * (0.08 / (1 - 0)) * 0 + (1 - .32) * (1 - .08) * (.55 / (1 - .32)), places=9)


class TestRace(unittest.TestCase):
    def test_brier_and_tail(self):
        p = rnd6([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02])
        oc = E.evaluate_record(rec(1, p), official(1), judge_for('2まくり'), LAB, BASE)
        r = oc['race']
        self.assertEqual(r['origin_hard'], '2まくり')
        self.assertAlmostEqual(r['brier_hard'], (0.5 - 1) ** 2 + sum(x * x for x in p[1:]))
        self.assertTrue(0.5 <= r['tail'] <= 1.0)  # 一番起きやすい結果が起きた → 他の合計 + u × 自分(同率の按分)
        self.assertAlmostEqual(r['tail'], 0.5 + r['tail_u'] * 0.5)
        self.assertFalse(r['unexpected'])
        oc = E.evaluate_record(rec(1, p), official(1), judge_for('その他'), LAB, BASE)
        self.assertAlmostEqual(oc['race']['tail'], oc['race']['tail_u'] * 0.02); self.assertTrue(oc['race']['unexpected'])

    def test_not_scored(self):
        p = rnd6(BASE)
        self.assertEqual(E.evaluate_record(rec(1, p), None, None, LAB, BASE)['race']['excluded_reason'], '公式結果なし')
        self.assertEqual(E.evaluate_record(rec(1, p), official(1, excluded=True), judge_for('その他'), LAB, BASE)['race']['excluded_reason'], '除外レース')
        self.assertEqual(E.evaluate_record(rec(1, p), official(1, course=(1, 3, 2, 4, 5, 6)), judge_for('その他'), LAB, BASE)['race']['excluded_reason'], '枠なりでない')
        self.assertEqual(E.evaluate_record(rec(1, p), official(1), None, LAB, BASE)['race']['excluded_reason'], '判定結果なし')

    def test_unknown_label_uses_fallback(self):
        j = judge_for('3まくり'); j['5'] = ['判断不可', '中']
        oc = E.evaluate_record(rec(1, rnd6(BASE)), official(1), j, LAB, BASE)
        self.assertEqual(oc['extra']['n_unknown_labels'], 1)

    def test_base_vs_base_is_zero(self):
        p = rnd6(BASE)
        oc = E.evaluate_record(rec(1, p), official(1), judge_for('3まくり', '中'), LAB, p)
        self.assertEqual(oc['race']['brier_soft'], oc['race']['brier_base_soft'])


class TestBoatClass(unittest.TestCase):
    ST = [{'boat': b, 'mean': 0.15, 'sd': 0.03} for b in range(1, 7)]

    def boats(self, success=None):
        return [{'boat': b, 'move_p': {'差し': .7, 'まくり': .2, 'まくり差し': .1} if b != 2 else {'差し': .8, 'まくり': .2, 'まくり差し': 0.0},
                 'success_p': success} for b in range(2, 7)]

    def cls(self, r, off, j):
        return {b['boat']: (b['class'], b['class_incl_low']) for b in E.evaluate_record(r, off, j, LAB, BASE)['boats']}

    def test_unexpected_soft(self):
        # 3号艇: まくりの予測確率 3% → 判定どおり「まくり」なら必ず想定外。ソフト期待値は「本当にまくりだった確率」に等しい
        p = rnd6(BASE)
        bs = self.boats(); bs[1]['move_p'] = {'差し': .7, 'まくり': .03, 'まくり差し': .27}
        for conf in ('高', '中', '低'):
            j = judge_for('3まくり', conf)
            oc = E.evaluate_record(rec(1, p, st=self.ST, boats=bs), official(1), j, LAB, p)
            x = next(b for b in oc['boats'] if b['boat'] == 3)
            sv = E.soft_move(3, 'まくり', conf, LAB, official(1))
            self.assertTrue(x['unexpected'])
            self.assertAlmostEqual(x['unexpected_soft'], sv[E.MOVES.index('まくり')], places=12)
        self.assertGreater(E.soft_move(3, 'まくり', '高', LAB, official(1))[1], E.soft_move(3, 'まくり', '低', LAB, official(1))[1])
        # 艇ごとの集計に件数とソフト期待件数が出る
        s = [(1, rec(1, p, st=self.ST, boats=bs), oc)]
        mu = E.boat_summary(s)['move_unexpected']
        self.assertEqual(mu[3]['count'], 1); self.assertAlmostEqual(mu[3]['soft_expected_count'], x['unexpected_soft'])

    def test_order(self):
        p = rnd6(BASE)
        j = judge_for('その他', '中')
        self.assertEqual(set(self.cls(rec(1, p, st=self.ST, boats=self.boats()), official(1, excluded=True), j).values()), {('E0', 'E0')})
        c = self.cls(rec(1, p, st=self.ST, boats=self.boats()), official(1, course=(1, 3, 2, 4, 5, 6)), j)
        self.assertEqual(c[2], ('E1', 'E1')); self.assertEqual(c[3], ('E1', 'E1')); self.assertEqual(c[4], ('当たり', '当たり'))
        st = {b: 0.15 for b in range(1, 7)}; st[4] = 0.25  # z = 3.33 > 1.96
        c = self.cls(rec(1, p, st=self.ST, boats=self.boats()), official(1, st=st), j)
        self.assertEqual(c[4], ('E2', 'E2')); self.assertEqual(c[5], ('当たり', '当たり'))
        j2 = dict(j); j2['5'] = ['まくり', '中']; j2['6'] = ['まくり', '低']
        c = self.cls(rec(1, p, st=self.ST, boats=self.boats()), official(1), j2)
        self.assertEqual(c[5], ('E3', 'E3')); self.assertEqual(c[6], ('E0', 'E3'))
        self.assertEqual(c[1], ('判定なし', '判定なし'))  # 1号艇は動きの予測を持たない

    def test_low_label_is_e0_first(self):
        # 設計書5.1: 確信度「低」は E0(評価対象外)。上から順なので、進入違いより先に E0 になる。低も含めた分類では E1
        p = rnd6(BASE); j = judge_for('その他', '中'); j['3'] = ['差し', '低']
        c = self.cls(rec(1, p, st=self.ST, boats=self.boats()), official(1, course=(1, 3, 2, 4, 5, 6)), j)
        self.assertEqual(c[3], ('E0', 'E1')); self.assertEqual(c[2], ('E1', 'E1'))

    def test_pending_is_not_e0(self):
        c = self.cls(rec(1, rnd6(BASE), boats=self.boats()), None, None)
        self.assertEqual(set(c.values()), {('判定なし', '判定なし')})
        c = self.cls(rec(1, rnd6(BASE), boats=self.boats()), official(1), None)
        self.assertEqual(c[3], ('判定なし', '判定なし'))

    def test_bad_judge_and_back(self):
        p = rnd6(BASE)
        for bad in ('判定失敗', [1, 2], 3):
            oc = E.evaluate_record(rec(1, p, boats=self.boats()), official(1), bad, LAB, BASE)
            self.assertEqual(oc['race']['excluded_reason'], '判定結果なし'); self.assertTrue(oc['extra']['pending'])
        for back in ([[1, None, 3, 2, 4, 5], '低'], ['1-3-2-4-5-6', '中'], [[], '中'], 'x'):
            self.assertIsNone(E.success_from_back(back, 3))

    def test_e4_inner_by_course(self):
        # 進入 1,2,4,3,5,6(予想どおり)。4号艇は3コースなので、内側は1・2号艇。外側の3号艇に負けても成功
        p = rnd6(BASE); crs = (1, 2, 4, 3, 5, 6)
        j = judge_for('その他', '中'); j['4'] = ['まくり', '中']; j['back'] = [[3, 4, 1, 2, 5, 6], '中']
        bs = self.boats({'差し': None, 'まくり': 0.9, 'まくり差し': None}); bs[2]['move_p'] = {'差し': .2, 'まくり': .7, 'まくり差し': .1}
        c = {b['boat']: b for b in E.evaluate_record(rec(1, p, boats=bs, courses=crs), official(1, course=crs), j, LAB, BASE)['boats']}
        self.assertEqual(c[4]['class'], '当たり'); self.assertTrue(c[4]['success_checked'])

    def test_e4(self):
        p = rnd6(BASE)
        j = judge_for('その他', '中'); j['back'] = [[1, 3, 2, 4, 5, 6], '中']  # 3号艇は1・2号艇の前 → 成功
        sp = {'差し': 0.2, 'まくり': 0.5, 'まくり差し': None}  # 差しは失敗と予測
        c = {b['boat']: b for b in E.evaluate_record(rec(1, p, boats=self.boats(sp)), official(1), j, LAB, BASE)['boats']}
        self.assertEqual(c[3]['class'], '当たり')  # 1号艇の前に出ていないので失敗 = 予測どおり
        self.assertTrue(c[3]['success_checked'])
        j['back'] = [[3, 1, 2, 4, 5, 6], '中']
        c = {b['boat']: b for b in E.evaluate_record(rec(1, p, boats=self.boats(sp)), official(1), j, LAB, BASE)['boats']}
        self.assertEqual(c[3]['class'], 'E4')

    def test_e4_without_back(self):
        c = {b['boat']: b for b in E.evaluate_record(rec(1, rnd6(BASE), boats=self.boats({'差し': .5, 'まくり': .5, 'まくり差し': .5})),
                                                     official(1), judge_for('その他', '中'), LAB, BASE)['boats']}
        self.assertEqual(c[3]['class'], '当たり'); self.assertFalse(c[3]['success_checked'])

    def test_success_from_back(self):
        self.assertTrue(E.success_from_back([[4, 1, 2, 3, 5, 6], '中'], 4))
        self.assertFalse(E.success_from_back([[1, 4, 2, 3, 5, 6], '中'], 4))
        self.assertIsNone(E.success_from_back([[4, 2, 3], '中'], 4))  # 1号艇が並びに無い
        self.assertIsNone(E.success_from_back(None, 4))


class TestStats(unittest.TestCase):
    def test_crit_matches_gate(self):
        # GATE 第4章: 300件なら22件以上で過信、600件なら40件以上
        self.assertEqual(E.crit_unexpected(300), 22); self.assertEqual(E.crit_unexpected(600), 40)
        self.assertAlmostEqual(E.binom_sf(22, 300, 0.05), 0.0486, places=4)
        self.assertAlmostEqual(E.binom_sf(21, 300, 0.05), 0.0776, places=4)

    def test_wilson(self):
        lo, hi = E.wilson(50, 100); self.assertAlmostEqual(lo, 0.4038, places=3); self.assertAlmostEqual(hi, 0.5962, places=3)

    def test_bss_identical(self):
        b = [0.3, 0.2, 0.5, 0.1]
        r = E.bss_ci(b, b)
        self.assertEqual(r['bss'], 0.0); self.assertEqual(r['lo'], 0.0); self.assertEqual(r['hi'], 0.0)

    def test_bss_ci_normal_agree(self):
        rng = random.Random(1)
        base = [rng.uniform(0.1, 0.5) for _ in range(500)]; model = [x - rng.gauss(0.01, 0.05) for x in base]
        r = E.bss_ci(base, model)
        self.assertLess(abs(r['lo'] - r['lo_normal']), 0.01); self.assertLess(abs(r['hi'] - r['hi_normal']), 0.01)


    def test_bss_day_clusters(self):
        rng = random.Random(2)
        base = [rng.uniform(0.1, 0.5) for _ in range(400)]; model = [x - rng.gauss(0.01, 0.05) for x in base]
        days = ['d%02d' % (i // 12) for i in range(400)]
        r1 = E.bss_ci(base, model); r2 = E.bss_ci(base, model, days)
        self.assertEqual(r1['bss'], r2['bss']); self.assertNotEqual((r1['lo'], r1['hi']), (r2['lo'], r2['hi']))

    def test_crit_alpha10(self):
        self.assertLessEqual(E.binom_sf(E.crit_unexpected(300, 0.10), 300, 0.10), 0.05)
        self.assertGreater(E.binom_sf(E.crit_unexpected(300, 0.10) - 1, 300, 0.10), 0.05)


class TestSimulation(unittest.TestCase):
    """答えが分かっている模擬データで、評価の仕組みそのものを確かめる(S3 の目的)"""
    N = 6000

    def sim(self, make_p, truth_p=None, seed=0, mark=lambda p: '○'):
        rng = random.Random(seed); out = []
        for i in range(self.N):
            p = rnd6(make_p(rng))
            t = draw(truth_p(p) if truth_p else p, rng)
            r = rec(i, p, mark=mark(p)); r['race']['race_id'] = '24_20260910_01'
            out.append((i + 1, r, E.evaluate_record(r, official(i), judge_for(OC[t]), LAB, rnd6(BASE))))
        return out

    def rate_ok(self, k, n, p=0.05):
        # 99.9%の範囲(正規近似)に入るか
        sd = math.sqrt(n * p * (1 - p)); return abs(k - n * p) < 3.29 * sd

    def test_calibrated_unexpected_is_5pct(self):
        s = self.sim(lambda r: dirichlet(r, 7, 0.7))
        k = sum(oc['race']['unexpected'] for _, _, oc in s)
        self.assertTrue(self.rate_ok(k, self.N), k)

    def test_ties_are_split_by_u(self):
        # 同率の起点がある予測(4まくりと5まくりが同じ)でも、按分の乱数で想定外がちょうど約5%になる
        tie = [0.30, 0.30, 0.02, 0.02, 0.02, 0.30, 0.04]
        s = self.sim(lambda r: tie)
        k = sum(oc['race']['unexpected'] for _, _, oc in s)
        self.assertTrue(self.rate_ok(k, self.N), k)
        u = [oc['race']['tail_u'] for _, _, oc in s]
        self.assertLess(abs(sum(u) / len(u) - 0.5), 0.02)

    def test_overconfident_is_detected(self):
        sharpen = lambda p: [x ** 3 / sum(y ** 3 for y in p) for x in p]
        s = self.sim(lambda r: sharpen(dirichlet(r, 7, 0.7)), truth_p=lambda p: [x ** (1 / 3) / sum(y ** (1 / 3) for y in p) for x in p])
        k = sum(oc['race']['unexpected'] for _, _, oc in s)
        self.assertGreater(k, E.crit_unexpected(self.N), k)

    def test_informative_model_beats_base(self):
        # 真の確率を知るモデル vs 平均の割合(基準)。BSS > 0 で区間の下限も 0 を超える
        s = self.sim(lambda r: dirichlet(r, 7, 0.7))
        bm = [oc['race']['brier_hard'] for _, _, oc in s]
        mean_p = rnd6([sum(r['origin']['p'][c] for _, r, _ in s) / self.N for c in range(7)])
        bb = [P.brier(mean_p, [1.0 if oc['race']['origin_hard'] == OC[c] else 0.0 for c in range(7)]) for _, _, oc in s]
        r = E.bss_ci(bb, bm, B=300)
        self.assertGreater(r['lo'], 0)

    def test_summary_and_gate(self):
        # 予測の自信度で◎/△を分け、◎の方が当たるか・帯が較正されているかを判定できる
        def mk(r):
            return dirichlet(r, 7, 0.5)
        mark = lambda p: '◎' if max(p) >= 0.5 else ('△' if max(p) < 0.3 else '○')
        s = self.sim(mk, mark=mark)
        sel = E.select_scored(s, {'eval_from': '2026-09-01T00:00:00+09:00'})
        self.assertEqual(len(sel), self.N)
        rep = E.summarize(sel)
        self.assertGreater(rep['bands']['◎']['hit_hard'], rep['bands']['△']['hit_hard'])
        self.assertTrue(rep['unexpected']['ok'])
        g = E.gate(sel)
        self.assertEqual(g[0]['n'], 300); self.assertTrue(g[0]['reached'])
        # 較正済みのモデルでも、300件目では偶然 22件(判定値ちょうど)になり条件2で落ちる(誤って落とす確率は約5%)。
        # 600件目の2回目では合格する。2回だけ見る決まりの動きをそのまま確かめる
        self.assertEqual((g[0]['cond2_unexpected'], g[0]['cond2_hard']), (22, False))
        self.assertEqual(g[0]['cond2'], g[0]['cond2_soft'] if E.UNEXPECTED_COUNT == 'soft' else g[0]['cond2_hard'])
        self.assertTrue(g[1]['cond2_hard']); self.assertEqual(len(g), 2 if not g[0]['pass'] else 1)
        # 判定どおりの件数で数える設定なら、300件目で落ちて600件目で合格する
        old = E.UNEXPECTED_COUNT
        try:
            E.UNEXPECTED_COUNT = 'hard'
            g = E.gate(sel)
            self.assertEqual((g[0]['cond2'], g[0]['pass']), (False, False))
            self.assertTrue(g[1]['pass']); self.assertEqual(len(g), 2)
        finally:
            E.UNEXPECTED_COUNT = old

    def test_boat_main_follows_low_setting(self):
        # 確信度「低」を主集計に含める設定(既定)では、艇ごとの内訳を class_incl_low で数える
        s = self.sim(lambda r: dirichlet(r, 7, 0.7))[:20]
        old = E.LOW_IN_MAIN
        try:
            for flag, key in ((True, 'class_incl_low'), (False, 'class')):
                E.LOW_IN_MAIN = flag
                b = E.boat_summary(s)
                self.assertEqual(b['main'], key)
                tot = collections.Counter()
                for _, _, oc in s:
                    for x in oc['boats']: tot[x[key]] += 1
                self.assertEqual(sum((collections.Counter(v) for v in b['by_boat'].values()), collections.Counter()), tot)
        finally:
            E.LOW_IN_MAIN = old

    def test_gate_uses_ledger_order_and_two_looks(self):
        s = self.sim(lambda r: dirichlet(r, 7, 0.7))[:450]
        shuffled = list(reversed(s))
        sel = E.select_scored(shuffled, {'eval_from': '2026-09-01T00:00:00+09:00'})
        self.assertEqual([x[0] for x in sel[:3]], [1, 2, 3])
        g = E.gate(sel)
        self.assertTrue(g[0]['reached']); self.assertFalse(g[1]['reached']) if len(g) > 1 else None

    def test_gate_holds_when_result_pending(self):
        # 台帳の5番目の結果・判定がまだ無いときは、300件目の判定を保留する(届いた後に別の300件で判定し直せないように)
        s = self.sim(lambda r: dirichlet(r, 7, 0.7))[:320]
        seq, r, oc = s[4]
        s[4] = (seq, r, E.evaluate_record(r, official(4), None, LAB, rnd6(BASE)))
        g = E.gate(E.eligible(s, {'eval_from': '2026-09-01T00:00:00+09:00'}))
        self.assertEqual(g, [{'n': 300, 'reached': False, 'held': True, 'pending_seq': 5}])
        # 除外レース(結果が確定して対象外)は保留にせず飛ばす
        s[4] = (seq, r, E.evaluate_record(r, official(4, excluded=True), judge_for('その他'), LAB, rnd6(BASE)))
        g = E.gate(E.eligible(s, {'eval_from': '2026-09-01T00:00:00+09:00'}))
        self.assertTrue(g[0]['reached']); self.assertEqual(g[0]['last_seq'], 301)

    def test_select_scored_rules(self):
        p = rnd6(BASE)
        a = (1, rec(1, p, mode='backtest'), E.evaluate_record(rec(1, p), official(1), judge_for('その他'), LAB, BASE))
        b = (2, rec(2, p, created='2026-08-31T14:59:59+00:00'), E.evaluate_record(rec(2, p), official(2), judge_for('その他'), LAB, BASE))
        c = (3, rec(3, p, created='2026-08-31T15:00:00+00:00'), E.evaluate_record(rec(3, p), official(3), judge_for('その他'), LAB, BASE))
        d = (4, rec(4, p), E.evaluate_record(rec(4, p), official(4, excluded=True), judge_for('その他'), LAB, BASE))
        sel = E.select_scored([a, b, c, d], {'eval_from': '2026-09-01T00:00:00+09:00'})
        self.assertEqual([x[0] for x in sel], [3])  # 本番・eval_from(日本時間)以降・除外なし
        self.assertEqual([x[0] for x in E.select_scored([a, b, c, d], {'eval_from': None}, include_backtest=True)], [1, 2, 3])
        self.assertEqual(E.select_scored([a, b, c, d], {'eval_from': None}), [])

    def test_st_misread_is_5pct_when_calibrated(self):
        rng = random.Random(5); n = 4000; k = 0
        stp = [{'boat': b, 'mean': 0.15, 'sd': 0.04} for b in range(1, 7)]
        for i in range(n):
            st = {b: rng.gauss(0.15, 0.04) for b in range(1, 7)}
            oc = E.evaluate_record(rec(i, rnd6(BASE), st=stp), official(i, st=st), judge_for('その他', '中'), LAB, BASE)
            k += sum(b['class'] == 'E2' for b in oc['boats'])
        self.assertTrue(self.rate_ok(k, n * 6), k)


class TestEndToEnd(unittest.TestCase):
    """predrec で記録を作って保存 → evaluate.run で採点するまでを通す"""
    def setUp(self):
        self.root = tempfile.mkdtemp()
        shutil.copytree(os.path.join(HERE, 'schema'), os.path.join(self.root, 'schema'))
        os.makedirs(os.path.join(self.root, 'models')); os.makedirs(os.path.join(self.root, 'versions'))
        with open(os.path.join(self.root, 'models', 'm.py'), 'w', encoding='utf-8') as f:
            f.write("def predict(s):\n    return {'origin': %r, 'premise': {'entry': {'courses': [1,2,3,4,5,6], 'p_as_predicted': None}, 'st': None, 'form': None},"
                    " 'scenario_id': None, 'boats': None, 'boat1': None, 'rationale': []}\n" % dict(zip(OC, BASE)))
        for v in ('base-t', 'model-t'):
            man = {'version': v, 'purpose': 't', 'model': 't', 'frozen_at': '2026-10-07T13:00:00+09:00', 'built_from': {}, 'eval_from': None,
                   'origin_classes': OC, 'bands': {'lo': 0.2, 'hi': 0.5}, 'bands_method': 't', 'success_definition': None,
                   'escape_definition': None, 'soft_label_table': 'test', 'entry': 'models/m.py', 'code_files': ['models/m.py'],
                   'code_files_hash': P.code_files_hash(['models/m.py'], self.root), 'notes': []}
            with open(os.path.join(self.root, 'versions', v + '.json'), 'w', encoding='utf-8') as f: json.dump(man, f, ensure_ascii=False)
        e = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')
        for a in (['init', '-q'], ['add', '-A'], ['commit', '-qm', 'i']):
            subprocess.run(['git', '-C', self.root] + a, check=True, capture_output=True, env=e)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_run(self):
        man = P.load_manifest(self.root, 'model-t')
        res, jd = [], {}
        for i in range(1, 6):
            rid = '24_20260910_%02d' % i
            r, s = P.make_record({'race_id': rid}, man, rid, None, 'backtest', code_root=self.root)
            P.save(r, s, self.root)
            off = official(i - 1); off['rno'] = i; res.append(off)
            jd[rid] = judge_for(['3まくり', 'その他', '2まくり', '3まくり差し', '6まくり'][i - 1], '中')
        res[1]['excluded'] = True
        rp = os.path.join(self.root, 'res.jsonl'); jp = os.path.join(self.root, 'judge.json'); lp = os.path.join(self.root, 'lab.json')
        with open(rp, 'w', encoding='utf-8') as f: f.write('\n'.join(json.dumps(x, ensure_ascii=False) for x in res))
        for p_, o in ((jp, jd), (lp, LAB)):
            with open(p_, 'w', encoding='utf-8') as f: json.dump(o, f, ensure_ascii=False)
        rep = E.run(self.root, 'model-t', 'base-t', rp, jp, lp, include_backtest=True, write_outcomes=True)
        self.assertEqual(rep['n_records'], 5)
        self.assertEqual(rep['summary']['n'], 4)
        self.assertEqual(rep['n_by_reason'], {'数える': 4, '除外レース': 1})
        self.assertEqual(rep['summary']['brier_soft']['bss_race']['bss'], 0.0)  # 基準と同じモデル
        self.assertEqual(rep['summary']['unexpected']['count'], 1)  # 6まくり(2.4%)だけ
        self.assertTrue(os.path.exists(os.path.join(self.root, 'outcomes', '24', '20260910', '24_20260910_05__model-t.outcome.json')))
        self.assertEqual(rep['coverage']['n_target'], 4); self.assertEqual(rep['coverage']['n_missing'], 0)
        oc = json.load(open(os.path.join(self.root, 'outcomes', '24', '20260910', '24_20260910_01__model-t.outcome.json'), encoding='utf-8'))
        P.check_schema(oc, 'outcome')
        self.assertFalse(oc['race']['scored'])  # バックテストは判定に数えない
        rep2 = E.run(self.root, 'model-t', 'base-t', rp, jp, lp)
        self.assertEqual(rep2['summary']['n'], 0)  # バックテストは判定に数えない
        self.assertEqual(rep2['n_by_reason'], {'本番でない・eval_from より前': 4, '除外レース': 1})
        self.assertEqual(rep2['boats']['class'], {})  # 本番の記録が無いので外れ分類の内訳も空

    def files(self):
        rp = os.path.join(self.root, 'res.jsonl'); jp = os.path.join(self.root, 'judge.json'); lp = os.path.join(self.root, 'lab.json')
        with open(rp, 'w', encoding='utf-8') as f: f.write(json.dumps(dict(official(0), rno=1), ensure_ascii=False))
        for p_, o in ((jp, {'24_20260910_01': judge_for('3まくり', '中')}), (lp, LAB)):
            with open(p_, 'w', encoding='utf-8') as f: json.dump(o, f, ensure_ascii=False)
        return rp, jp, lp

    def save_one(self):
        man = P.load_manifest(self.root, 'model-t')
        r, s = P.make_record({'race_id': '24_20260910_01'}, man, '24_20260910_01', None, 'backtest', code_root=self.root)
        return P.save(r, s, self.root)

    def test_tampered_record_is_refused(self):
        rp_ = self.save_one(); rp, jp, lp = self.files()
        path = os.path.join(self.root, rp_); d = json.load(open(path, encoding='utf-8'))
        d['origin']['p'][0], d['origin']['p'][1] = d['origin']['p'][1], d['origin']['p'][0]
        json.dump(d, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
        with self.assertRaises(E.EvalError): E.run(self.root, 'model-t', 'base-t', rp, jp, lp, include_backtest=True)

    def test_changed_eval_from_is_refused(self):
        self.save_one(); rp, jp, lp = self.files()
        vp = os.path.join(self.root, 'versions', 'model-t.json'); m = json.load(open(vp, encoding='utf-8'))
        m['eval_from'] = '2026-09-01T00:00:00+09:00'; json.dump(m, open(vp, 'w', encoding='utf-8'), ensure_ascii=False)
        with self.assertRaises((E.EvalError, P.RecordError)): E.run(self.root, 'model-t', 'base-t', rp, jp, lp)

    def test_missing_labels_file(self):
        self.save_one(); rp, jp, _ = self.files()
        import io, contextlib
        argv = ['evaluate.py', 'report', '--root', self.root, '--version', 'model-t', '--base-version', 'base-t',
                '--results', rp, '--judge', jp, '--labels', os.path.join(self.root, 'nope.json'), '--include-backtest']
        with mock_argv(argv), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(E.main(), 2)


if __name__ == '__main__':
    unittest.main()
