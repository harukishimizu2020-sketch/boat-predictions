"""予測記録の評価器(S3)。予測記録(records/)を、レース後の公式結果と判定AIの結果に照らして採点し、長期の成績をまとめる。

設計: DESIGN_prediction_verification.md 第4章・5.1・5.2・5.4・5.5 / GATE_stage2to3_20261007.md / README.md(このフォルダ)。
使い方:
  python evaluate.py report --root . --version <版> --base-version <基準の版> --results results_all.jsonl --judge judge.json
                            [--labels labels/soft-v1.json] [--include-backtest] [--write-outcomes] [--out report.json]
決まり:
  - 予測記録は書き換えない。採点結果(outcomes/)は、記録・公式結果・判定結果から毎回同じ値に作り直せる派生ファイル。
  - 段階2→3の判定に数えるのは、同じ版の本番(live)記録で、版の eval_from 以降に作られ、枠なり・除外なしのレースだけ。
    数える順番は台帳(ledger.jsonl)の順。300件目・600件目で区切る(GATE 第3章)。
  - 想定外の同率按分の乱数は predrec.tail_u(記録のハッシュとレース後の公式結果から決まる)。
  - 標準ライブラリだけで動く(端末に numpy が無くても動かせるように)。
"""
import argparse, collections, datetime as dt, hashlib, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predrec as pr

FORMAT = 'outcome-1'
OC = pr.ORIGIN_CLASSES
MOVES = pr.MOVES
ALPHA = pr.ALPHA
GATE_N = (300, 600)        # GATE 第3章: 判定は300件と600件の2回だけ
# 評価の決め方(S3_REPORT.md 6章)。2026-10-07 ユーザー決定: 条件2はソフトの期待件数で数える・確信度「低」も主集計に含める
UNEXPECTED_COUNT = 'soft'  # 条件2の数え方。'soft': ソフト正解の期待件数を判定値と比べる / 'hard': 判定どおりの件数(GATE の文字どおり)
LOW_IN_MAIN = True         # 艇ごとの外れ分類の主集計に確信度「低」を含めるか(含めない場合は設計書4章どおり E0)
GATE_Z = 1.959963984540054  # 各回 片側2.5%(両側95%区間の下限 > 0)
BOOT_B = 2000               # ブートストラップの回数
BOOT_SEED = 20261007


class EvalError(Exception):
    pass


# ---------- 判定ラベル → 正解の確率の内訳(ソフトラベル。設計書5.5) ----------
def load_labels(path):
    with open(path, encoding='utf-8') as f:
        lab = json.load(f)
    for k in ('version', 'moves', 'high_self', 'table', 'fallback'):
        if k not in lab: raise EvalError('ラベルの表に %s がありません: %s' % (k, path))
    if lab['moves'] != MOVES: raise EvalError('ラベルの表の動きの並びが違います')
    for key, d in lab['table'].items():
        if abs(sum(d.get(m, 0.0) for m in MOVES) - 1) > 1e-6: raise EvalError('ラベルの表 %s の合計が1ではありません' % key)
    for b, v in lab['fallback'].items():
        if len(v) != 3 or abs(sum(v) - 1) > 1e-6: raise EvalError('判断不可の置き換え %s 号艇の合計が1ではありません' % b)
    return lab


def soft_move(boat, t, c, lab, official=None):
    """2〜6号艇の判定ラベル(動きタイプ t, 確信度 c)を [差し, まくり, まくり差し] の確率に直す。判断不可・欠損は None。
    kyotei_mark1/pipeline/a5_common.py の soft(t, c, b, r)(2026-10-07 修正後)と同じ計算:
      1) 判定AIには着順と決まり手を渡しているので、1着艇(1号艇以外)で決まり手と同じラベルは確定(rules.winner_kimarite_certain)
      2) 確信度「高」は 自分 high_self・他 (1-high_self)/2
      3) 2号艇は table2 にあればそれを使う。無ければ table を使い、まくり差しの分を差しに移す(rules.boat2_makurizashi_to_sashi)"""
    if t not in MOVES: return None
    rules = lab.get('rules', {})
    if rules.get('winner_kimarite_certain') and official is not None and boat != 1:
        order = official.get('order') or []
        if order and int(order[0]) == boat and official.get('kimarite') == t:
            return [1.0 if m == t else 0.0 for m in MOVES]
    if c == '高':
        hs = lab['high_self']; v = [hs if m == t else round((1 - hs) / 2, 12) for m in MOVES]
    elif boat == 2 and '%s|%s' % (t, c) in lab.get('table2', {}):
        d = lab['table2']['%s|%s' % (t, c)]; return [d.get(m, 0.0) for m in MOVES]
    else:
        d = lab['table'].get('%s|%s' % (t, c), {t: 1.0}); v = [d.get(m, 0.0) for m in MOVES]
    if rules.get('boat2_makurizashi_to_sashi') and boat == 2:
        v = [v[0] + v[2], v[1], 0.0]  # 2号艇の内側は1号艇だけなので、まくり差しは起きない
    return v


def origin_soft(P):
    """P: {艇番(2〜6): [差し, まくり, まくり差し]}。艇ごとのラベル誤差を独立とみなして起点7分類の確率を出す(GATE 第2章)"""
    out = collections.defaultdict(float); none_m = 1.0
    for b in range(2, 7):
        pm = P[b][1]; out['%dまくり' % b] += none_m * pm; none_m *= (1 - pm)
    rest = none_m; none_ms = 1.0
    for b in range(2, 7):
        q = P[b][2] / max(1e-9, 1 - P[b][1])
        out['%dまくり差し' % b] += rest * none_ms * q; none_ms *= (1 - q)
    out['全艇差し'] += rest * none_ms
    v = [0.0] * len(OC)
    for k, p in out.items(): v[OC.index(k) if k in OC else OC.index('その他')] += p
    s = sum(v)
    return [x / s for x in v]


def origin_hard(H):
    """H: {艇番: 動きタイプ}。一番内側でまくりに行った艇、無ければ一番内側でまくり差しに行った艇"""
    for b in range(2, 7):
        if H.get(b) == 'まくり': k = '%dまくり' % b; return k if k in OC else 'その他'
    for b in range(2, 7):
        if H.get(b) == 'まくり差し': k = '%dまくり差し' % b; return k if k in OC else 'その他'
    return 'その他'


# ---------- 公式結果 ----------
def race_valid(off):
    """(数えるか, 理由)。分析と同じ: 除外(進入変更・F・出遅れ・欠場など)が無く、枠なり進入"""
    if off is None: return False, '公式結果なし'
    if off.get('excluded'): return False, '除外レース'
    if off.get('course') != [1, 2, 3, 4, 5, 6]: return False, '枠なりでない'
    return True, None


def actual_st(off, boat):
    for e in off.get('entry') or []:
        if int(e.get('boat', 0)) == boat:
            try: return float(str(e.get('st')).replace('F', '-').replace('L', ''))
            except (TypeError, ValueError): return None
    return None


def actual_course(off, boat):
    c = off.get('course') or []
    return c.index(boat) + 1 if boat in c else None


def judge_label(judge_race, boat):
    v = judge_race.get(str(boat)) if isinstance(judge_race, dict) else None
    return (v[0], v[1]) if isinstance(v, list) and len(v) >= 2 else (None, None)


def success_from_back(back, boat, course=None):
    """1マーク直後の並び [[先頭→最後尾], 確信度] で、自分より内側のコースの艇すべての前に出たか。
    course: 実際の進入(コース1〜6の艇番)。並びが無い・崩れている・内側の艇が並びに無いときは None"""
    if not back or not isinstance(back, list) or not isinstance(back[0], list) or not back[0]: return None
    try: order = [int(x) for x in back[0]]
    except (TypeError, ValueError): return None
    course = list(course) if course else [1, 2, 3, 4, 5, 6]
    if boat not in order or boat not in course: return None
    inner = course[:course.index(boat)]
    if any(b not in order for b in inner): return None
    return all(order.index(boat) < order.index(b) for b in inner)


def boat_u(rhash, official, boat):
    """艇ごとの動きの想定外の按分に使う乱数(predrec.tail_u と同じ作り方で、艇番を足して別の値にする)"""
    return pr.tail_u('%s#%d' % (rhash, boat), official)


# ---------- 1レースの採点 ----------
def evaluate_record(rec, official, judge_race, lab, base_p, evaluated_at=None):
    rh = rec['integrity']['record_hash']
    p = rec['origin']['p']
    if not isinstance(judge_race, dict): judge_race = None
    ok, why = race_valid(official)
    race = {'scored': False, 'excluded_reason': why, 'origin_hard': None, 'origin_soft': None,
            'brier_soft': None, 'brier_hard': None, 'brier_base_soft': None, 'brier_base_hard': None,
            'tail': None, 'tail_u': None, 'unexpected': None}
    extra = {'pending': official is None or (ok and judge_race is None), 'n_unknown_labels': None, 'n_low_labels': None,
             'soft_unexpected': None, 'soft_unexpected_10': None, 'unexpected_10': None, 'hit_soft': None, 'hit_hard': None}
    if ok and judge_race is None: ok, why = False, '判定結果なし'; race['excluded_reason'] = why
    if ok:
        P, H, n_unk = {}, {}, 0
        for b in range(2, 7):
            t, c = judge_label(judge_race, b)
            v = soft_move(b, t, c, lab, official)
            if v is None: v = list(lab['fallback'][str(b)]); n_unk += 1
            P[b] = v; H[b] = t
        q = origin_soft(P); h = origin_hard(H); hi = OC.index(h)
        y = [1.0 if i == hi else 0.0 for i in range(len(OC))]
        u = pr.tail_u(rh, official)
        tl = pr.tail(p, hi, u)
        mi = OC.index(rec['main_scenario']['origin'])
        race.update({'origin_hard': h, 'origin_soft': [round(x, 6) for x in q],
                     'brier_soft': pr.brier(p, q), 'brier_hard': pr.brier(p, y),
                     'brier_base_soft': pr.brier(base_p, q), 'brier_base_hard': pr.brier(base_p, y),
                     'tail': tl, 'tail_u': u, 'unexpected': tl < ALPHA})
        extra.update({'n_unknown_labels': n_unk, 'n_low_labels': sum(judge_label(judge_race, b)[1] == '低' for b in range(2, 7)),
                      'unexpected_10': tl < 0.10,
                      'soft_unexpected_10': sum(q[c] * (pr.tail(p, c, u) < 0.10) for c in range(len(OC))),
                      # ソフト正解での想定外の期待値: 起点 c が真である確率 q_c で、c が起きた場合の想定外を平均
                      'soft_unexpected': sum(q[c] * (pr.tail(p, c, u) < ALPHA) for c in range(len(OC))),
                      'hit_soft': q[mi], 'hit_hard': 1.0 if mi == hi else 0.0})
    boats = [evaluate_boat(rec, official, judge_race, lab, b) for b in range(1, 7)]
    out = {'format': FORMAT, 'race_id': rec['race']['race_id'], 'version': rec['model']['version'], 'record_hash': rh,
           'evaluated_at': evaluated_at or pr.now_utc(),
           'official': None if official is None else {
               'course': official.get('course'), 'st': {str(b): actual_st(official, b) for b in range(1, 7)},
               'order': official.get('order'), 'kimarite': official.get('kimarite'),
               'flags': official.get('flags') or [], 'excluded': bool(official.get('excluded'))},
           'judge': {'source': str((judge_race or {}).get('_source', '')), 'moves': {str(b): list(judge_label(judge_race, b)) for b in range(1, 7)},
                     'back': (judge_race or {}).get('back'), 'soft_label_table': lab['version']},
           'race': race, 'boats': boats, 'extra': extra}
    return out


def evaluate_boat(rec, official, judge_race, lab, b):
    """設計書5.1 の外れ分類を上から順に当てはめる。
    class: 主集計(確信度「低」のラベルは E0)。class_incl_low: 低も含めた場合(設計書第4章: 両方を並べる)。
    結果や判定がまだ無い艇は「判定なし」(E0 には入れない)"""
    r = {'boat': b, 'class': None, 'class_incl_low': None, 'st_z': None, 'move_tail': None, 'unexpected': None,
         'unexpected_soft': None, 'match_soft': None, 'label': None, 'success_checked': False}
    def done(cls, cls_low=None):
        r['class'] = cls; r['class_incl_low'] = cls if cls_low is None else cls_low; return r
    if official is None: return done('判定なし')
    if official.get('excluded'): return done('E0')
    t, c = judge_label(judge_race, b)
    r['label'] = [t, c]
    low = c == '低'
    pc = rec['premise']['entry']['courses']
    if actual_course(official, b) != (pc.index(b) + 1): return done('E0' if low else 'E1', 'E1')
    st = rec['premise'].get('st')
    if st is not None:
        s = next(x for x in st if x['boat'] == b)
        a = actual_st(official, b)
        if a is not None:
            r['st_z'] = pr.st_z(s['mean'], s['sd'], a)
            if pr.st_misread(r['st_z']): return done('E0' if low else 'E2', 'E2')
    pb = None if rec.get('boats') is None or b == 1 else next(x for x in rec['boats'] if x['boat'] == b)
    if pb is None or judge_race is None: return done('判定なし')  # 1号艇の攻められ方の照合は未実装(GATE 8.3)
    if t not in MOVES: return done('E0')
    mp = [pb['move_p'][m] for m in MOVES]
    ti = MOVES.index(t)
    bu = boat_u(rec['integrity']['record_hash'], official, b)
    r['move_tail'] = pr.tail(mp, ti, bu)
    r['unexpected'] = r['move_tail'] < ALPHA
    pred = max(range(3), key=lambda k: (mp[k], -k))
    sv = soft_move(b, t, c, lab, official)
    # ソフト正解での想定外の期待値(起点の soft_expected_count と同じ考え方): 動き m が本当だった確率 × m なら想定外か
    r['unexpected_soft'] = sum(sv[m] * (pr.tail(mp, m, bu) < ALPHA) for m in range(3))
    r['match_soft'] = sv[pred]  # 設計書5.5: 予測した動きが本当に起きた確率(ソフトラベルでの一致の期待値)
    if pred != ti: cls = 'E3'
    else:
        cls = '当たり'
        sp = pb.get('success_p')
        ok = success_from_back(judge_race.get('back'), b, official.get('course'))
        if sp is not None and sp.get(t) is not None and ok is not None:
            r['success_checked'] = True
            if (sp[t] >= 0.5) != ok: cls = 'E4'
    return done('E0' if low else cls, cls)


# ---------- 統計 ----------
def mean(x):
    return sum(x) / len(x) if x else float('nan')


def sd(x):
    if len(x) < 2: return float('nan')
    m = mean(x); return math.sqrt(sum((a - m) ** 2 for a in x) / (len(x) - 1))


def binom_sf(k, n, p):
    """P(X >= k), X ~ Bin(n, p)"""
    if k <= 0: return 1.0
    if k > n: return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                        for i in range(k, n + 1)))


def crit_unexpected(n, p0=ALPHA, a=0.05):
    """GATE 第4章: P(X >= k) <= a となる最小の k。想定外がこの件数以上なら過信"""
    for k in range(n + 2):
        if binom_sf(k, n, p0) <= a: return k
    return n + 1


def wilson(k, n, z=1.959963984540054):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bss_ci(base, model, clusters=None, B=BOOT_B, seed=BOOT_SEED):
    """BSS = (平均Brier基準 − 平均Brierモデル)/ 平均Brier基準 と、その95%区間(ブートストラップの百分位)。
    clusters を渡すと、そのまとまり(開催日)ごとに選び直す"""
    n = len(base)
    if n == 0: return {'n': 0, 'bss': None, 'lo': None, 'hi': None}
    est = (sum(base) - sum(model)) / sum(base)
    rng = random.Random(seed)
    if clusters is None:
        groups = [[i] for i in range(n)]
    else:
        g = collections.OrderedDict()
        for i, c in enumerate(clusters): g.setdefault(c, []).append(i)
        groups = list(g.values())
    vals = []
    for _ in range(B):
        sb = sm = 0.0
        for _ in range(len(groups)):
            for i in groups[rng.randrange(len(groups))]:
                sb += base[i]; sm += model[i]
        vals.append((sb - sm) / sb if sb > 0 else 0.0)
    vals.sort()
    lo = vals[int(math.floor(0.025 * (B - 1)))]; hi = vals[int(math.ceil(0.975 * (B - 1)))]
    d = [a - b for a, b in zip(base, model)]
    se = sd(d) / math.sqrt(n) if n > 1 else float('nan')
    mb = mean(base)
    return {'n': n, 'bss': est, 'lo': lo, 'hi': hi, 'z': (mean(d) / se) if se and se > 0 else None,
            'lo_normal': (mean(d) - GATE_Z * se) / mb if n > 1 else None, 'hi_normal': (mean(d) + GATE_Z * se) / mb if n > 1 else None}


def two_mean_z(a, b):
    """2群の平均の差の z(両側)。0/1でもソフト得点でも使える"""
    if len(a) < 2 or len(b) < 2: return None, None
    se = math.sqrt(sd(a) ** 2 / len(a) + sd(b) ** 2 / len(b))
    if se == 0: return None, None
    z = (mean(a) - mean(b)) / se
    return z, 2 * (1 - norm_cdf(abs(z)))


# ---------- 集計 ----------
def eligible(pairs, manifest, include_backtest=False):
    """pairs: [(seq, rec, outcome)]。判定の対象になりうる記録(本番・eval_from 以降)を台帳の順に返す。
    include_backtest=True なら全記録(過去レースでの試し。判定には使わない)"""
    out = []
    ef = manifest.get('eval_from')
    for seq, rec, oc in sorted(pairs, key=lambda x: x[0]):
        if not include_backtest:
            if rec['mode'] != 'live': continue
            if ef is None or pr.parse_time(rec['created_at']) < pr.parse_time(ef): continue
        out.append((seq, rec, oc))
    return out


def select_scored(pairs, manifest, include_backtest=False):
    """判定に数えるもの(対象の記録のうち、枠なり・除外なしで、結果と判定がそろったレース)を台帳の順に返す"""
    return [x for x in eligible(pairs, manifest, include_backtest) if x[2]['race']['excluded_reason'] is None]


def summarize(sel, label='全件'):
    """選ばれたレース(台帳の順)の成績"""
    n = len(sel)
    R = [oc['race'] for _, _, oc in sel]; X = [oc['extra'] for _, _, oc in sel]
    rep = {'label': label, 'n': n}
    if n == 0: return rep
    days = [rec['race']['hd'] for _, rec, _ in sel]
    for kind in ('soft', 'hard'):
        bm = [r['brier_' + kind] for r in R]; bb = [r['brier_base_' + kind] for r in R]
        rep['brier_' + kind] = {'model': mean(bm), 'base': mean(bb), 'bss_race': bss_ci(bb, bm), 'bss_day': bss_ci(bb, bm, days)}
    k = sum(1 for r in R if r['unexpected']); ks = sum(x['soft_unexpected'] for x in X)
    rep['unexpected'] = {'count': k, 'rate': k / n, 'crit': crit_unexpected(n), 'ok': k < crit_unexpected(n),
                         'p_value': binom_sf(k, n, ALPHA), 'soft_expected_count': ks, 'soft_ok': ks < crit_unexpected(n),
                         'u_mean': mean([r['tail_u'] for r in R]),
                         # 設計書5.4: 参考として α=10% も並べる
                         'alpha10': {'count': sum(1 for x in X if x['unexpected_10']), 'soft_expected_count': sum(x['soft_unexpected_10'] for x in X),
                                     'crit': crit_unexpected(n, 0.10)}}
    bands = {}
    for m in ('◎', '○', '△'):
        g = [(rec, oc) for _, rec, oc in sel if rec['confidence']['mark'] == m]
        hs = [oc['extra']['hit_soft'] for _, oc in g]; hh = [oc['extra']['hit_hard'] for _, oc in g]
        pm = [rec['main_scenario']['p'] for rec, _ in g]
        if not g: bands[m] = {'n': 0}; continue
        se = sd(hs) / math.sqrt(len(hs)) if len(hs) > 1 else float('nan')
        lo, hi = mean(hs) - 1.96 * se, mean(hs) + 1.96 * se
        bands[m] = {'n': len(g), 'p_mean': mean(pm), 'hit_soft': mean(hs), 'hit_soft_ci': [lo, hi],
                    'hit_hard': mean(hh), 'hit_hard_ci': list(wilson(sum(hh), len(hh))),
                    'calibrated_soft': (lo <= mean(pm) <= hi) if len(hs) > 1 else None}
    z, pv = two_mean_z([oc['extra']['hit_soft'] for _, rec, oc in sel if rec['confidence']['mark'] == '◎'],
                       [oc['extra']['hit_soft'] for _, rec, oc in sel if rec['confidence']['mark'] == '△'])
    rep['bands'] = bands
    rep['bands_test'] = {'z': z, 'p_two_sided': pv,
                         'ok': None if z is None else (z > 0 and pv < 0.05 and all(bands[m].get('calibrated_soft') is not False for m in bands if bands[m]['n'])),
                         'note': None if z is not None else '◎または△が2件未満で比べられない(全レースが同じ帯など)'}
    # 較正: 起点7分類の確率を0.1刻みの帯に分け、帯ごとに予測の平均と実際(ソフト・ハード)の発生率を比べる
    cal = [{'bin': '%.1f-%.1f' % (i / 10, (i + 1) / 10), 'n': 0, 'p': 0.0, 'soft': 0.0, 'hard': 0.0} for i in range(10)]
    for _, rec, oc in sel:
        for c in range(len(OC)):
            pc = rec['origin']['p'][c]; i = min(9, int(pc * 10))
            cal[i]['n'] += 1; cal[i]['p'] += pc; cal[i]['soft'] += oc['race']['origin_soft'][c]
            cal[i]['hard'] += 1.0 if oc['race']['origin_hard'] == OC[c] else 0.0
    rep['calibration'] = [dict(b, p=b['p'] / b['n'], soft=b['soft'] / b['n'], hard=b['hard'] / b['n']) for b in cal if b['n']]
    rep['origin_freq'] = {'soft': [mean([oc['race']['origin_soft'][c] for _, _, oc in sel]) for c in range(len(OC))],
                          'hard': [mean([1.0 if oc['race']['origin_hard'] == OC[c] else 0.0 for _, _, oc in sel]) for c in range(len(OC))]}
    rep['n_unknown_labels'] = sum(x['n_unknown_labels'] for x in X)
    return rep


def boat_summary(pairs):
    """外れ分類の内訳(艇ごと。設計書5.2「外れ分類の内訳」)、STの読み違いの率、動きの想定外の率、ソフトラベルでの一致の期待値"""
    key = 'class_incl_low' if LOW_IN_MAIN else 'class'
    cnt = {k: collections.Counter() for k in ('class', 'class_incl_low')}
    by_boat = collections.defaultdict(collections.Counter)
    z_all = collections.defaultdict(list); unexp = collections.defaultdict(list); ms = collections.defaultdict(list)
    unexp_s = collections.defaultdict(list)
    for _, rec, oc in pairs:
        for b in oc['boats']:
            cnt['class'][b['class']] += 1; cnt['class_incl_low'][b['class_incl_low']] += 1
            by_boat[b['boat']][b[key]] += 1
            if b['st_z'] is not None: z_all[b['boat']].append(b['st_z'])
            if b['unexpected'] is not None and b[key] != 'E0':
                unexp[b['boat']].append(b['unexpected']); unexp_s[b['boat']].append(b.get('unexpected_soft'))
            if b.get('match_soft') is not None and b[key] != 'E0': ms[b['boat']].append(b['match_soft'])
    e2 = {b: {'n': len(v), 'rate_outside_95': sum(abs(z) > pr.E2_Z for z in v) / len(v), 'z_mean': mean(v), 'z_sd': sd(v)}
          for b, v in sorted(z_all.items())}
    mu = {b: {'n': len(v), 'count': sum(v), 'rate': sum(v) / len(v), 'crit': crit_unexpected(len(v)),
              'soft_expected_count': (sum(unexp_s[b]) if None not in unexp_s[b] else None)} for b, v in sorted(unexp.items())}
    return {'main': key, 'class': dict(cnt['class']), 'class_incl_low': dict(cnt['class_incl_low']),
            'by_boat': {b: dict(c) for b, c in sorted(by_boat.items())}, 'st_misread': e2, 'move_unexpected': mu,
            'match_soft_mean': {b: mean(v) for b, v in sorted(ms.items())}}


def gate(elig):
    """GATE 第0章の3条件。台帳の順で数えて300件目・600件目の2回だけ判定する(多重検定を避けるため、それ以外の件数では判定しない)。
    elig: 対象の記録(台帳の順)。n件目までの間に、結果か判定がまだ届いていない記録が1件でもあれば、その回は保留する
    (届いた後に中身の違う n 件で判定し直せないように)"""
    out = []
    for n in GATE_N:
        sel = []; pending = None
        for x in elig:
            if len(sel) >= n: break
            if x[2]['extra']['pending']: pending = x[0]; break
            if x[2]['race']['excluded_reason'] is None: sel.append(x)
        if pending is not None:
            out.append({'n': n, 'reached': False, 'held': True, 'pending_seq': pending}); break
        if len(sel) < n:
            out.append({'n': n, 'reached': False, 'held': False}); break
        s = summarize(sel, '先頭%d件' % n)
        b = s['brier_soft']['bss_race']
        c1 = b['lo'] is not None and b['lo'] > 0
        c2h = s['unexpected']['ok']; c2s = s['unexpected']['soft_ok']
        c2 = c2s if UNEXPECTED_COUNT == 'soft' else c2h
        c3 = s['bands_test']['ok']
        out.append({'n': n, 'reached': True, 'last_seq': sel[-1][0], 'cond1_bss_lo': b['lo'], 'cond1': c1,
                    'cond2_unexpected': s['unexpected']['count'], 'cond2_soft_expected': s['unexpected']['soft_expected_count'],
                    'cond2_crit': s['unexpected']['crit'], 'cond2': c2, 'cond2_hard': c2h, 'cond2_soft': c2s,
                    'cond2_count': UNEXPECTED_COUNT,
                    'cond3': c3, 'pass': bool(c1 and c2 and c3),
                    'note': '条件2は %s で判定(cond2_count)。もう一方は参考(2026-10-07 ユーザー決定)' % ('ソフト正解の期待件数' if UNEXPECTED_COUNT == 'soft' else '判定どおりの件数')})
        if out[-1]['pass']: break
    return out


# ---------- 読み込みと実行 ----------
def load_results(path):
    res = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            res['%s_%s_%02d' % (r['jcd'], r['hd'], int(r['rno']))] = r
    return res


def load_judge(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_records(root, version):
    """台帳の順に (seq, 記録) を返す。台帳と記録の食い違いは predrec.verify で先に確かめておくこと"""
    out = []
    for line in pr._read_ledger(root):
        if line.get('version') != version: continue
        rp, _ = pr.paths_for(line['race_id'], version)
        out.append((line['seq'], pr._load(os.path.join(root, rp))))
    return out


def load_input(root, rec):
    _, ip = pr.paths_for(rec['race']['race_id'], rec['model']['version'])
    return pr._load(os.path.join(root, ip))


def coverage(root, pairs, manifest, results, include_backtest):
    """記録漏れ(README 第4章: 自信のあるレースだけ記録することへの対策)。
    対象の期間(eval_from の日から最後の記録の日まで)の大村の枠なり・除外なしのレースのうち、記録が無いもの"""
    el = eligible(pairs, manifest, include_backtest)
    if not el: return {'n_target': 0, 'n_missing': 0, 'rate': None}
    ef = manifest.get('eval_from')
    start = (pr.parse_time(ef) + pr.JST).strftime('%Y%m%d') if (ef and not include_backtest) else min(r['race']['hd'] for _, r, _ in el)
    end = max(r['race']['hd'] for _, r, _ in el)
    have = {r['race']['race_id'] for _, r, _ in el}
    target = sorted(k for k, o in results.items() if str(o.get('jcd')) == '24' and start <= o['hd'] <= end and race_valid(o)[0])
    miss = [k for k in target if k not in have]
    return {'from': start, 'to': end, 'n_target': len(target), 'n_missing': len(miss), 'rate': len(miss) / len(target) if target else None,
            'missing': miss[:50]}


def manifest_first_commit(root, version):
    """版の設定ファイルが最初に git にコミットされた時刻(両張りの確認用: eval_from より前に公開されているか)。
    コミット時刻は PC で偽装できるので、正式には GitHub に push が届いた時刻で確かめる(README 第4章)"""
    try: t = pr._git(root, 'log', '--diff-filter=A', '--format=%cI', '--', 'versions/%s.json' % version).split()
    except Exception: return None
    return t[-1] if t else None


def run(root, version, base_version, results, judge, labels_path, include_backtest=False, write_outcomes=False, verify=True):
    if verify:
        probs = pr.verify(root, replay=True)
        if probs: raise EvalError('記録の検証で問題が見つかったため採点しません:\n' + '\n'.join(probs))
    manifest = pr.load_manifest(root, version)
    base_man = pr.load_manifest(root, base_version)
    lab = load_labels(labels_path)
    res = load_results(results); jd = load_judge(judge)
    pairs = []
    for seq, rec in load_records(root, version):
        pr.check_record(rec, manifest)  # 版の設定が記録時と同じか(manifest_hash)・記録のハッシュ
        rid = rec['race']['race_id']
        # 基準: 同じレースの同じ入力で基準の版のモデルを動かした起点の確率(README 第5章)
        base_p = pr.normalize_output(pr.run_model(base_man, load_input(root, rec), root))['origin']
        pairs.append((seq, rec, evaluate_record(rec, res.get(rid), jd.get(rid), lab, base_p)))
    sel = select_scored(pairs, manifest, include_backtest)
    chosen = {s for s, _, _ in sel}
    for s, rec, oc in pairs:
        oc['race']['scored'] = s in chosen and not include_backtest
    if write_outcomes:
        for _, rec, oc in pairs:
            p = os.path.join(root, 'outcomes', '24', rec['race']['hd'], '%s__%s.outcome.json' % (rec['race']['race_id'], version))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w', encoding='utf-8', newline='\n') as f: json.dump(oc, f, ensure_ascii=False, indent=1)
    why = collections.Counter(oc['race']['excluded_reason'] or ('数える' if s in chosen else '本番でない・eval_from より前')
                              for s, _, oc in pairs)
    el = eligible(pairs, manifest, include_backtest)
    low_free = [x for x in sel if x[2]['extra']['n_low_labels'] == 0]
    return {'version': version, 'base_version': base_version, 'labels': lab['version'],
            'include_backtest': include_backtest, 'n_records': len(pairs), 'n_by_reason': dict(why),
            'coverage': coverage(root, pairs, manifest, res, include_backtest),
            'manifest_first_commit': manifest_first_commit(root, version), 'eval_from': manifest.get('eval_from'),
            'summary': summarize(sel), 'summary_without_low': summarize(low_free, '確信度「低」のラベルを含むレースを除く'),
            'boats': boat_summary(el), 'gate': gate(el) if not include_backtest else 'バックテストは判定に使わない'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('report')
    r.add_argument('--root', default='.'); r.add_argument('--version', required=True); r.add_argument('--base-version', required=True)
    r.add_argument('--results', required=True); r.add_argument('--judge', required=True)
    r.add_argument('--labels', default=os.path.join(HERE, 'labels', 'soft-v1.json'))
    r.add_argument('--include-backtest', action='store_true'); r.add_argument('--write-outcomes', action='store_true')
    r.add_argument('--out')
    a = ap.parse_args()
    try:
        rep = run(a.root, a.version, a.base_version, a.results, a.judge, a.labels, a.include_backtest, a.write_outcomes)
    except (EvalError, pr.RecordError, OSError, ValueError) as e:
        print('エラー:', e, file=sys.stderr); return 2
    s = json.dumps(rep, ensure_ascii=False, indent=1, default=str)
    if a.out:
        with open(a.out, 'w', encoding='utf-8', newline='\n') as f: f.write(s)
    print(s)
    return 0


if __name__ == '__main__':
    sys.exit(main())
