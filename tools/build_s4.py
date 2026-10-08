"""S4 本モデル s4-v1 の特徴づくり・交差検証・学習・凍結(設計: S4_DESIGN.md 第2章)。

使い方:
  python tools/build_s4.py --results <results_all.jsonl> --extract <_gate_extract.json> --results-long <results_omura_long.jsonl> \
                           [--log <進捗ファイル>] [--out <交差検証の結果 JSON>] [--write]
  --write を付けると models/s4_v1_params.json・versions/s4-v1.json・reports/s4/cv_result.json を新規に書く(既にあれば止まる)。

決まり:
  - 正解: 判定AIのラベルを、大村は labels/soft-v2.json、他会場は labels/soft-v1.json(evaluate.soft_move、規則つき)で確率に直す。
    判断不可・欠損は表の fallback(構築期間の同じコースの平均)。起点は evaluate.origin_soft で合成。
  - 学習: 2026-09-01 より前の、判定済み・出走表あり・枠なり・除外なしの全会場のレース(戸田 02 は使わない)。
  - 特徴: models/s4_v1.py の features()(学習と予測で同じ関数)。履歴の特徴はその日より前の判定済みレースだけから作る。
  - 候補 A(起点7分類の多項ロジスティック)と B(艇ごとの動き3分類 → origin_soft)。どちらも L2 正則化(α)と、基準と混ぜる λ を
    構築期間(大村 2026-07-02〜08-31)の中だけで、日付順に区切った交差検証(学習は常に検証ブロックの初日より前)の
    ソフト Brier で選ぶ。λ=0(基準そのもの)も候補。評価期間(2026-09-01〜)は見ない。
  - 交差検証の基準は「検証ブロックより前の大村の判定済みレースの平均」(基準の作り方と同じで、未来を見ない)。
  - 凍結時の基準は baseline-v2(models/baseline_v2.py の ORIGIN・MOVE)。
  - numpy だけで動く(sklearn・scipy は使わない)。多項ロジスティックは L-BFGS を自前で実装。
"""
import argparse, bisect, collections, datetime as dt, importlib.util, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import predrec as pr
import evaluate as E

OC = pr.ORIGIN_CLASSES; MOVES = pr.MOVES; K = len(OC)
OMURA = '24'; TODA = '02'
CON_FROM, CON_TO, SPLIT = '20260702', '20260831', '20260901'   # 構築期間と、学習の終わり(評価期間の始まり)
PRIOR_BEFORE = CON_FROM     # 履歴特徴の事前割合は構築期間より前のデータだけから(定数として凍結)
ST_FROM = '20251215'        # ST の式は大村の 2025-12-15〜2026-08-31(baseline と同じ期間)
HIST_K = 5.0                # 履歴の割合を事前へ縮める強さ(件数換算)
ST_N = 20; ST_MIN_N = 5     # 過去20走のST、5走未満なら使わない
N_FOLDS = 5
ALPHAS = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003]   # 最初の試走で α=1(端)が選ばれたので、より強い正則化 3・10 を足した(評価期間は見ていない)
LAMBDAS = [round(0.1 * i, 1) for i in range(11)]
VERSION = 's4-v1'
LOG = None


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        with open(LOG, 'a', encoding='utf-8') as f: f.write(s + '\n')


def load_model_module():
    spec = importlib.util.spec_from_file_location('s4_v1_model', os.path.join(HERE, 'models', 's4_v1.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ---------- データ ----------
def race_key(r):
    return '%s_%s_%02d' % (r['jcd'], r['hd'], int(r['rno']))


def st_value(s):
    """本番ST: F は負、L・欠損は None"""
    s = str(s or '')
    if not s or s.startswith('L'): return None
    try: return -float(s[1:]) if s.startswith('F') else float(s)
    except ValueError: return None


class Data:
    def __init__(self, results, extract, results_long, lab_omura, lab_other):
        self.res = E.load_results(results)
        with open(extract, encoding='utf-8') as f: ex = json.load(f)
        self.judge = ex['judge']; self.aux = ex['aux_all']
        self.lab_omura = E.load_labels(lab_omura); self.lab_other = E.load_labels(lab_other)
        # ST の履歴: 全会場(戸田を除く)の公式結果 + 大村の長期の結果。除外レースも含む(F は負の値)
        self.st_hist = collections.defaultdict(list)
        rows = list(self.res.values())
        if results_long:
            with open(results_long, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if race_key(r) not in self.res: rows.append(r)
        for r in rows:
            if str(r['jcd']) == TODA or not r.get('tobans'): continue
            for e in r.get('entry') or []:
                t = r['tobans'].get(str(e.get('boat')))
                v = st_value(e.get('st'))
                if t and v is not None: self.st_hist[t].append((r['hd'], int(r['rno']), v))
        for v in self.st_hist.values(): v.sort()
        self.st_dates = {t: [x[0] for x in v] for t, v in self.st_hist.items()}
        # 判定済み・枠なり・除外なし(戸田を除く)のレースのソフトラベル
        self.labels = {}   # key -> {'P': {b: vec}, 'known': {b: bool}, 'q': origin soft, 'hard': origin hard}
        self.move_hist = collections.defaultdict(list)  # 登番 -> [(hd, course, vec)](ラベルが判断不可でない艇だけ)
        for k in sorted(self.judge):
            r = self.res.get(k)
            if r is None or str(r['jcd']) == TODA or not E.race_valid(r)[0]: continue
            lab = self.lab_omura if str(r['jcd']) == OMURA else self.lab_other
            P, known, H = {}, {}, {}
            for b in range(2, 7):
                t, c = E.judge_label(self.judge[k], b)
                v = E.soft_move(b, t, c, lab, r)
                known[b] = v is not None
                P[b] = v if v is not None else list(lab['fallback'][str(b)]); H[b] = t
                if v is not None:
                    tb = (r.get('tobans') or {}).get(str(b))
                    if tb: self.move_hist[tb].append((r['hd'], b, v))
            self.labels[k] = {'P': P, 'known': known, 'q': E.origin_soft(P), 'hard': OC.index(E.origin_hard(H)), 'jcd': str(r['jcd']), 'hd': r['hd']}
        for v in self.move_hist.values(): v.sort(key=lambda x: x[0])
        self.move_dates = {t: [x[0] for x in v] for t, v in self.move_hist.items()}

    def history(self, toban, b, hd):
        """その日(hd)より前の判定済みレースからの、この選手の動きの合計と、過去20走のST。未来は見ない"""
        out = {'same_course': {'n': 0.0, '差し': 0.0, 'まくり': 0.0, 'まくり差し': 0.0},
               'any_course': {'n': 0.0, '差し': 0.0, 'まくり': 0.0, 'まくり差し': 0.0}, 'st': {'n': 0, 'mean': None}}
        if toban:
            mh = self.move_hist.get(toban, [])
            for x in mh[:bisect.bisect_left(self.move_dates.get(toban, []), hd)]:
                for tgt in ((out['same_course'],) if x[1] == b else ()) + (out['any_course'],):
                    tgt['n'] += 1
                    for m, p in zip(MOVES, x[2]): tgt[m] += p
            sh = self.st_hist.get(toban, [])
            past = sh[:bisect.bisect_left(self.st_dates.get(toban, []), hd)][-ST_N:]
            if past: out['st'] = {'n': len(past), 'mean': sum(x[2] for x in past) / len(past)}
        for d in (out['same_course'], out['any_course']):
            for m in MOVES: d[m] = round(d[m], 6)
        if out['st']['mean'] is not None: out['st']['mean'] = round(out['st']['mean'], 6)
        return out

    def snapshot(self, k):
        """レース k のスナップショット(レース前の情報 + その日より前の履歴)。出走表が無いレースは card などが None"""
        r = self.res[k]; a = self.aux.get(k) or {}
        tobans = r.get('tobans') or {}
        return {'race_id': k, 'jcd': str(r['jcd']), 'hd': r['hd'], 'rno': int(r['rno']), 'captured_at': None,
                'tobans': {str(b): tobans.get(str(b)) for b in range(1, 7)},
                'card': a.get('card'), 'before': a.get('before'), 'original_tenjis': a.get('originalTenjis'),
                'race_class': a.get('raceClass'), 'title': a.get('title'),
                'history': {str(b): self.history(tobans.get(str(b)), b, r['hd']) for b in range(1, 7)},
                'history_cutoff': r['hd'], 'note': 'history はこの日より前の判定済みレース・公式結果だけから作った。オッズ・本番STは入れない'}

    def training_keys(self):
        return sorted(k for k in self.labels if k in self.aux and self.labels[k]['hd'] < SPLIT)


def feature_params(data):
    """履歴の事前割合(構築期間より前の判定済みレース、全会場)と会場の一覧"""
    acc = {b: np.zeros(3) for b in range(2, 7)}; n = {b: 0 for b in range(2, 7)}
    for k, L in data.labels.items():
        if L['hd'] >= PRIOR_BEFORE: continue
        for b in range(2, 7):
            if L['known'][b]: acc[b] += L['P'][b]; n[b] += 1
    prior = {str(b): [round(float(x), 6) for x in acc[b] / max(1, n[b])] for b in range(2, 7)}
    tot = sum(acc.values()); prior_any = [round(float(x), 6) for x in tot / max(1, sum(n.values()))]
    venues = sorted({data.labels[k]['jcd'] for k in data.training_keys()})
    return {'hist_prior': prior, 'hist_prior_any': prior_any, 'hist_k': HIST_K, 'venues': venues,
            'prior_n': {str(b): n[b] for b in range(2, 7)}, 'prior_period': '〜%s' % PRIOR_BEFORE}


# ---------- 多項ロジスティック(ソフト正解、L2、L-BFGS) ----------
def lbfgs(f_g, x0, max_iter=500, m=10, tol=1e-6):
    x = x0.copy(); f, g = f_g(x); S, Y = [], []
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol: break
        q = g.copy(); al = []
        for s, y in reversed(list(zip(S, Y))):
            a = (s @ q) / (y @ s); al.append(a); q -= a * y
        if S: q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        for (s, y), a in zip(zip(S, Y), reversed(al)):
            q += s * (a - (y @ q) / (y @ s))
        d = -q
        if g @ d >= 0: d = -g
        t = 1.0; gd = g @ d
        while True:
            xn = x + t * d; fn, gn = f_g(xn)
            if fn <= f + 1e-4 * t * gd or t < 1e-12: break
            t *= 0.5
        s = xn - x; y = gn - g
        if y @ s > 1e-12:
            S.append(s); Y.append(y)
            if len(S) > m: S.pop(0); Y.pop(0)
        if abs(f - fn) < 1e-12 * max(1.0, abs(f)): x, f, g = xn, fn, gn; break
        x, f, g = xn, fn, gn
    return x


def fit_multinomial(X, Q, alpha):
    """X: n×d(標準化済み)、Q: n×K のソフト正解(行の合計1)。損失 = 平均交差エントロピー + α/2·‖W‖²(切片は罰しない)"""
    n, d = X.shape; Kc = Q.shape[1]

    def f_g(theta):
        W = theta[:d * Kc].reshape(d, Kc); b = theta[d * Kc:]
        Z = X @ W + b; Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        loss = -np.sum(Q * np.log(np.maximum(P, 1e-300))) / n + 0.5 * alpha * np.sum(W * W)
        G = X.T @ (P - Q) / n + alpha * W
        return loss, np.r_[G.ravel(), (P - Q).sum(0) / n]
    th = lbfgs(f_g, np.zeros(d * Kc + Kc))
    W = th[:d * Kc].reshape(d, Kc); b = th[d * Kc:]
    return {'coef': W.T.tolist(), 'intercept': b.tolist()}


def predict_proba(X, model):
    W = np.array(model['coef']).T; b = np.array(model['intercept'])
    Z = X @ W + b; Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)


def scaler(Xraw):
    """欠損(NaN)は中央値、標準化は平均・標準偏差(学習データだけから)"""
    med = np.nanmedian(Xraw, 0); med = np.where(np.isnan(med), 0.0, med)
    X = np.where(np.isnan(Xraw), med, Xraw)
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd > 0, sd, 1.0)
    return {'impute_median': med.tolist(), 'scale_mean': mu.tolist(), 'scale_sd': sd.tolist()}


def transform(Xraw, sc):
    X = np.where(np.isnan(Xraw), np.array(sc['impute_median']), Xraw)
    sd = np.array(sc['scale_sd'])
    return (X - np.array(sc['scale_mean'])) / sd


def origin_from_moves(M):
    """M: {b: n×3} → n×7(evaluate.origin_soft をレースごとに)"""
    n = len(M[2])
    return np.array([E.origin_soft({b: list(M[b][i]) for b in range(2, 7)}) for i in range(n)])


def fit_all(X, rows, alpha):
    """候補 A(起点)と候補 B の部品(艇ごとの動き)を学習する"""
    Qo = np.array([r['q'] for r in rows])
    A = fit_multinomial(X, Qo, alpha)
    B = {}
    for b in range(2, 7):
        Qm = np.array([r['P'][b] for r in rows])
        if b == 2: Qm = Qm[:, :2] / Qm[:, :2].sum(1, keepdims=True)   # 2号艇はまくり差しが無い(2分類)
        B[b] = fit_multinomial(X, Qm, alpha)
    return A, B


def moves_proba(X, B):
    M = {}
    for b in range(2, 7):
        p = predict_proba(X, B[b])
        M[b] = np.c_[p, np.zeros(len(p))] if b == 2 else p
    return M


# ---------- 交差検証 ----------
def make_blocks(con_rows, n_folds):
    """構築期間のレースを日付順に n_folds 個の連続したブロックに分ける(件数がほぼ等しくなるように)"""
    dates = sorted({r['hd'] for r in con_rows}); cnt = collections.Counter(r['hd'] for r in con_rows)
    total = len(con_rows); blocks = []; cur = []; acc = 0
    for i, d in enumerate(dates):
        cur.append(d); acc += cnt[d]
        if acc >= total * (len(blocks) + 1) / n_folds and len(blocks) < n_folds - 1:
            blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    return blocks


def brier_rows(P, Y):
    return np.sum((P - Y) ** 2, 1)


def cross_validate(data, rows, fp, mod):
    """構築期間の大村のレースについて、日付順ブロックの交差検証で全候補の予測を集める。
    返り値: {config_key: {'origin': n×7, 'moves': {b: n×3}}} と、検証レースの並び・各レースの基準(そのブロックより前の大村の平均)"""
    con = [r for r in rows if r['jcd'] == OMURA and CON_FROM <= r['hd'] <= CON_TO]
    con.sort(key=lambda r: (r['hd'], r['key']))
    blocks = make_blocks(con, N_FOLDS)
    log('構築期間の大村 %d レース、%d ブロック: %s' % (len(con), len(blocks), [(b[0], b[-1], sum(1 for r in con if b[0] <= r['hd'] <= b[-1])) for b in blocks]))
    omura_all = [(L['hd'], L) for k, L in data.labels.items() if L['jcd'] == OMURA]
    preds = collections.defaultdict(lambda: {'origin': [], 'moves': {b: [] for b in range(2, 7)}})
    order = []; base_o = []; base_m = []; hard = []; q = []
    for bi, bl in enumerate(blocks):
        start = bl[0]
        tr = [r for r in rows if r['hd'] < start]
        va = [r for r in con if bl[0] <= r['hd'] <= bl[-1]]
        prior_o = [L['q'] for hd, L in omura_all if hd < start]
        prior_m = {b: [L['P'][b] for hd, L in omura_all if hd < start] for b in range(2, 7)}
        bo = np.mean(prior_o, 0); bm = {b: np.mean(prior_m[b], 0) for b in range(2, 7)}
        sc = scaler(np.array([r['x'] for r in tr], float))
        Xtr = transform(np.array([r['x'] for r in tr], float), sc); Xva = transform(np.array([r['x'] for r in va], float), sc)
        t0 = time.time()
        for alpha in ALPHAS:
            A, B = fit_all(Xtr, tr, alpha)
            pA = predict_proba(Xva, A); M = moves_proba(Xva, B); pB = origin_from_moves(M)
            for lam in LAMBDAS:
                for cand, pm in (('A', pA), ('B', pB)):
                    key = '%s|%g|%g' % (cand, alpha, lam)
                    preds[key]['origin'].append((1 - lam) * bo + lam * pm)
                    for b in range(2, 7): preds[key]['moves'][b].append((1 - lam) * bm[b] + lam * M[b])
        log('ブロック %d(%s〜%s): 学習 %d レース(大村 %d)、検証 %d レース、基準の学習元 大村 %d レース、%.0f 秒'
            % (bi + 1, bl[0], bl[-1], len(tr), sum(r['jcd'] == OMURA for r in tr), len(va), len(prior_o), time.time() - t0))
        order += [r['key'] for r in va]; base_o += [bo] * len(va); base_m += [bm] * len(va)
        hard += [r['hard'] for r in va]; q += [r['q'] for r in va]
    out = {k: {'origin': np.vstack(v['origin']), 'moves': {b: np.vstack(v['moves'][b]) for b in range(2, 7)}} for k, v in preds.items()}
    return out, order, np.array(base_o), base_m, np.array(hard), np.array(q), con


def summarize_cv(preds, order, base_o, base_m, hard, q, con):
    Yh = np.eye(K)[hard]
    Pm = {b: np.array([r['P'][b] for r in con]) for b in range(2, 7)}
    b_base = brier_rows(base_o, q); b_base_h = brier_rows(base_o, Yh)
    bm_base = {b: brier_rows(np.array([bm[b] for bm in base_m]), Pm[b]) for b in range(2, 7)}
    days = [r['hd'] for r in con]
    res = {}
    for key, v in preds.items():
        bs = brier_rows(v['origin'], q); bh = brier_rows(v['origin'], Yh)
        mv = {b: float(brier_rows(v['moves'][b], Pm[b]).mean()) for b in range(2, 7)}
        res[key] = {'brier_soft': float(bs.mean()), 'brier_hard': float(bh.mean()),
                    'bss_soft': float((b_base.mean() - bs.mean()) / b_base.mean()), 'bss_hard': float((b_base_h.mean() - bh.mean()) / b_base_h.mean()),
                    'move_brier': mv, 'move_brier_sum': float(sum(mv.values()))}
    base = {'brier_soft': float(b_base.mean()), 'brier_hard': float(b_base_h.mean()),
            'move_brier': {b: float(bm_base[b].mean()) for b in range(2, 7)}, 'move_brier_sum': float(sum(bm_base[b].mean() for b in range(2, 7)))}
    return res, base, b_base, days


def choose(res, base):
    """起点: 全候補(λ=0 を含む)の中でソフト Brier が最小。同点なら λ の小さい方・α の大きい(正則化の強い)方。
    動き: 候補 B の部品の動きの Brier の合計が最小(起点に B を選んだときはそれと同じ設定)"""
    def parse(k):
        c, a, l = k.split('|'); return c, float(a), float(l)
    items = [(round(v['brier_soft'], 9), parse(k)[2], -parse(k)[1], k) for k, v in res.items()]
    items.sort()
    best_key = items[0][3]
    cand, alpha, lam = parse(best_key)
    if lam == 0 or round(res[best_key]['brier_soft'], 9) >= round(base['brier_soft'], 9):
        origin = {'candidate': 'base', 'origin_alpha': None, 'origin_lambda': 0.0, 'cv_brier_soft': base['brier_soft']}
    else:
        origin = {'candidate': cand, 'origin_alpha': alpha, 'origin_lambda': lam, 'cv_brier_soft': res[best_key]['brier_soft']}
    mitems = [(round(v['move_brier_sum'], 9), parse(k)[2], -parse(k)[1], k) for k, v in res.items() if k.startswith('B|')]
    mitems.sort()
    mk = mitems[0][3]; _, ma, ml = parse(mk)
    if ml == 0 or round(res[mk]['move_brier_sum'], 9) >= round(base['move_brier_sum'], 9):
        move = {'move_alpha': None, 'move_lambda': 0.0, 'cv_move_brier_sum': base['move_brier_sum']}
    else:
        move = {'move_alpha': ma, 'move_lambda': ml, 'cv_move_brier_sum': res[mk]['move_brier_sum']}
    if origin['candidate'] == 'B':
        move = {'move_alpha': origin['origin_alpha'], 'move_lambda': origin['origin_lambda'],
                'cv_move_brier_sum': res[best_key]['move_brier_sum'], 'note': '起点に B を選んだので動きも同じ設定'}
    return dict(origin, **move)


# ---------- ST の式 ----------
def fit_st(data):
    """大村 ST_FROM〜CON_TO の枠なり・除外なしレース: ST = α_コース + β×(過去20走の平均ST − 中心)。sd は残差、sd0 は素の sd"""
    rows = []
    for k, r in data.res.items():
        if str(r['jcd']) != OMURA or not (ST_FROM <= r['hd'] <= CON_TO) or not E.race_valid(r)[0]: continue
        for b in range(1, 7):
            a = E.actual_st(r, b)
            if a is None: continue
            h = data.history((r.get('tobans') or {}).get(str(b)), b, r['hd'])['st']
            rows.append((b, a, h['mean'] if h['n'] >= ST_MIN_N else None))
    with_p = [x for x in rows if x[2] is not None]
    center = float(np.mean([x[2] for x in with_p]))
    Xd = np.array([[1.0 if x[0] == c else 0.0 for c in range(1, 7)] + [x[2] - center] for x in with_p])
    y = np.array([x[1] for x in with_p])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ coef
    st = {'alpha': {str(c): round(float(coef[c - 1]), 6) for c in range(1, 7)}, 'beta': round(float(coef[6]), 6), 'center': round(center, 6),
          'sd': {str(c): round(float(np.std([e for x, e in zip(with_p, resid) if x[0] == c], ddof=1)), 6) for c in range(1, 7)},
          'sd0': {str(c): round(float(np.std([x[1] for x in rows if x[0] == c], ddof=1)), 6) for c in range(1, 7)},
          'min_n': ST_MIN_N, 'n_rows': len(rows), 'n_with_p20': len(with_p), 'period': '%s〜%s(大村・枠なり・除外なし)' % (ST_FROM, CON_TO),
          'rmse_with_p20': round(float(np.sqrt(np.mean(resid ** 2))), 6),
          'rmse_course_only': round(float(np.sqrt(np.mean([(x[1] - np.mean([z[1] for z in with_p if z[0] == x[0]])) ** 2 for x in with_p]))), 6)}
    return st


# ---------- 実行 ----------
def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True); ap.add_argument('--extract', required=True); ap.add_argument('--results-long')
    ap.add_argument('--labels-omura', default=os.path.join(HERE, 'labels', 'soft-v2.json'))
    ap.add_argument('--labels-other', default=os.path.join(HERE, 'labels', 'soft-v1.json'))
    ap.add_argument('--log'); ap.add_argument('--out'); ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    LOG = a.log
    t0 = time.time()
    data = Data(a.results, a.extract, a.results_long, a.labels_omura, a.labels_other)
    log('判定済み・枠なり・除外なし(戸田除く) %d レース、出走表あり %d レース' % (len(data.labels), sum(1 for k in data.labels if k in data.aux)))
    mod = load_model_module()
    fp = feature_params(data)
    log('履歴の事前割合', json.dumps(fp, ensure_ascii=False))
    rows = []
    names = None
    for k in data.training_keys():
        L = data.labels[k]
        nm, vals = mod.features(data.snapshot(k), fp)
        if names is None: names = nm
        rows.append({'key': k, 'jcd': L['jcd'], 'hd': L['hd'], 'x': [np.nan if v is None else v for v in vals], 'q': L['q'], 'P': L['P'], 'hard': L['hard']})
    log('学習レース %d(大村 %d)、特徴 %d 個、%.0f 秒' % (len(rows), sum(r['jcd'] == OMURA for r in rows), len(names), time.time() - t0))
    Xraw = np.array([r['x'] for r in rows], float)
    log('欠損の多い特徴(上位10)', sorted(((float(np.isnan(Xraw[:, i]).mean()), n) for i, n in enumerate(names)), reverse=True)[:10])
    preds, order, base_o, base_m, hard, q, con = cross_validate(data, rows, fp, mod)
    res, base, b_base, days = summarize_cv(preds, order, base_o, base_m, hard, q, con)
    ch = choose(res, base)
    log('交差検証 基準: ソフト Brier %.5f ハード %.5f 動き合計 %.5f' % (base['brier_soft'], base['brier_hard'], base['move_brier_sum']))
    for cand in ('A', 'B'):
        for alpha in ALPHAS:
            line = ['%s α=%g:' % (cand, alpha)]
            for lam in LAMBDAS:
                v = res['%s|%g|%g' % (cand, alpha, lam)]; line.append('λ%.1f %.5f(%+.2f%%)' % (lam, v['brier_soft'], 100 * v['bss_soft']))
            log(' '.join(line))
    log('選んだ設定', json.dumps(ch, ensure_ascii=False))
    key = None if ch['candidate'] == 'base' else '%s|%g|%g' % (ch['candidate'], ch['origin_alpha'], ch['origin_lambda'])
    Pcv = preds[key]['origin'] if key else base_o
    bs = brier_rows(Pcv, q)
    ci_race = E.bss_ci(list(b_base), list(bs)); ci_day = E.bss_ci(list(b_base), list(bs), days)
    pm = Pcv.max(1)
    bands = {'lo': round(float(np.percentile(pm, 100 / 3)), 6), 'hi': round(float(np.percentile(pm, 200 / 3)), 6)} if key else {'lo': 0.0, 'hi': 1.01}
    per_block = []
    blocks = make_blocks(con, N_FOLDS)
    for bl in blocks:
        idx = [i for i, r in enumerate(con) if bl[0] <= r['hd'] <= bl[-1]]
        per_block.append({'from': bl[0], 'to': bl[-1], 'n': len(idx), 'brier_base': float(b_base[idx].mean()), 'brier_model': float(bs[idx].mean()),
                          'bss': float((b_base[idx].mean() - bs[idx].mean()) / b_base[idx].mean())})
    cv_summary = {'chosen': ch, 'bss_race': ci_race, 'bss_day': ci_day, 'bands': bands, 'n_val': len(order), 'per_block': per_block,
                  'main_p_quantiles': [float(np.percentile(pm, p)) for p in (0, 10, 25, 50, 75, 90, 100)]}
    log('ブロックごと', json.dumps(per_block, ensure_ascii=False))
    log('選んだ設定の交差検証 BSS(ソフト) レース単位 %.4f(%.4f〜%.4f) 日単位 %.4f(%.4f〜%.4f) 帯 %s'
        % (ci_race['bss'], ci_race['lo'], ci_race['hi'], ci_day['bss'], ci_day['lo'], ci_day['hi'], bands))
    st = fit_st(data)
    log('ST の式', json.dumps(st, ensure_ascii=False))
    cv_out = {'created_at': pr.now_utc(), 'version': VERSION, 'n_train': len(rows), 'n_train_omura': sum(r['jcd'] == OMURA for r in rows),
              'feature_names': names, 'feature_params': fp, 'alphas': ALPHAS, 'lambdas': LAMBDAS, 'n_folds': N_FOLDS,
              'cv_base': base, 'cv_results': res, 'cv_summary': cv_summary, 'val_order': order, 'st': st,
              'cv_pred_chosen': [[round(float(x), 6) for x in p] for p in Pcv]}
    if a.out:
        with open(a.out, 'w', encoding='utf-8', newline='\n') as f: json.dump(cv_out, f, ensure_ascii=False, indent=1)
    if not a.write: return
    # ---- 凍結: 学習データ全部(< SPLIT)で学習し、基準は baseline-v2 ----
    spec = importlib.util.spec_from_file_location('baseline_v2', os.path.join(HERE, 'models', 'baseline_v2.py'))
    bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
    sc = scaler(Xraw); X = transform(Xraw, sc)
    params = {'version': VERSION, 'feature_names': names, 'feature_params': fp, **sc,
              'base_origin': dict(bl.ORIGIN), 'base_move': {str(b): dict(bl.MOVE[b]) for b in range(2, 7)},
              'choice': {k: ch[k] for k in ('candidate', 'origin_alpha', 'origin_lambda', 'move_alpha', 'move_lambda')},
              'st': {k: st[k] for k in ('alpha', 'beta', 'center', 'sd', 'sd0', 'min_n')},
              'origin_model': None, 'move_models': None}
    oa = ch['origin_alpha'] if ch['candidate'] == 'A' else None
    ma = ch['move_alpha']
    if oa is not None:
        params['origin_model'] = fit_multinomial(X, np.array([r['q'] for r in rows]), oa)
    if ma is not None:
        _, B = fit_all(X, rows, ma); params['move_models'] = {str(b): B[b] for b in range(2, 7)}
    else:
        # 動きのモデルを使わない(λ=0)ときも compute() が動くように、係数 0 の形だけ置く
        params['move_models'] = {str(b): {'coef': [[0.0] * len(names)] * (2 if b == 2 else 3), 'intercept': [0.0] * (2 if b == 2 else 3)} for b in range(2, 7)}
    if params['origin_model'] is None:
        params['origin_model'] = {'coef': [[0.0] * len(names)] * K, 'intercept': [0.0] * K}
    files = {'models/s4_v1_params.json': params}
    for p in list(files) + ['versions/%s.json' % VERSION, 'reports/s4/cv_result.json']:
        if os.path.exists(os.path.join(HERE, p)): sys.exit('既にあります(上書きしない): ' + p)
    for p, obj in files.items():
        with open(os.path.join(HERE, p), 'w', encoding='utf-8', newline='\n') as f: json.dump(obj, f, ensure_ascii=False, indent=0); f.write('\n')
    code_files = ['models/s4_v1.py', 'models/s4_v1_params.json']
    man = {'version': VERSION,
           'purpose': 'S4 本モデル(レース前の情報だけで起点7分類・動き・STを出す)。評価期間 2026-09-01〜10-01 の大村で baseline-v2 と比べる。この版の過去レースの記録は段階2→3の判定に使わない',
           'model': '候補 %s(%s)。混ぜる割合 λ: 起点 %.1f・動き %.1f。L2 α: 起点 %s・動き %s。特徴 %d 個(出走表・展示・風波・選手×コースの履歴)。ST はコース切片+β×過去20走の平均ST' % (
               ch['candidate'], {'A': '起点7分類の多項ロジスティック', 'B': '艇ごとの動き3分類→origin_soft', 'base': '基準そのもの(交差検証で λ=0 が最良)'}[ch['candidate']],
               ch['origin_lambda'], ch['move_lambda'], ch['origin_alpha'], ch['move_alpha'], len(names)),
           'frozen_at': pr.now_utc(),
           'built_from': {'data': 'kyotei_mark1/data/results_all.jsonl・analysis/_gate_extract.json(judge・aux_all)・results_omura_long.jsonl(STの履歴)',
                          'period': '学習 〜%s の判定済み・出走表あり・枠なり・除外なし %d レース(全会場、戸田除く。大村 %d)。候補と λ の選択は大村 %s〜%s の %d レースの日付順交差検証(%d ブロック)' % (
                              '20260831', len(rows), sum(r['jcd'] == OMURA for r in rows), CON_FROM, CON_TO, len(con), N_FOLDS),
                          'labels': '大村 labels/soft-v2.json、他会場 labels/soft-v1.json', 'base': 'baseline-v2',
                          'cv': {'base_brier_soft': base['brier_soft'], 'chosen': ch, 'bss_race': {k: ci_race[k] for k in ('bss', 'lo', 'hi')}, 'bss_day': {k: ci_day[k] for k in ('bss', 'lo', 'hi')}},
                          'cv_result_file': 'reports/s4/cv_result.json'},
           'eval_from': None, 'origin_classes': OC, 'bands': bands,
           'bands_method': '構築期間の交差検証の予測の主シナリオ確率の3分位(GATE 第5章)' if key else '基準そのもの(全レース同じ確率)のため帯を分けない',
           'success_definition': None, 'escape_definition': None, 'soft_label_table': 'labels/soft-v2.json',
           'entry': 'models/s4_v1.py', 'code_files': code_files, 'code_files_hash': pr.code_files_hash(code_files, HERE),
           'notes': ['評価期間(2026-09-01〜10-01)は候補・λ・特徴を決めるときに見ていない(S4_DESIGN.md 2.3)',
                     'soft-v2 の表は評価期間に含まれる手動判定40レースの照合を使っている(基準と本モデルに同じだけ効く。S4_DESIGN.md 第3章)',
                     'オッズ・本番STは使わない。履歴の特徴はその日より前の判定済みレースだけ', '成功確率・1号艇・足の評価は出さない(null)']}
    pr.check_manifest(man)
    with open(os.path.join(HERE, 'versions', '%s.json' % VERSION), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(man, f, ensure_ascii=False, indent=1); f.write('\n')
    os.makedirs(os.path.join(HERE, 'reports', 's4'), exist_ok=True)
    with open(os.path.join(HERE, 'reports', 's4', 'cv_result.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(cv_out, f, ensure_ascii=False, indent=1); f.write('\n')
    # 凍結したモデルが学習データで動くことと、pure Python の計算が numpy と合うことを確かめる
    mod2 = load_model_module()
    out = mod2.predict(data.snapshot(rows[-1]['key']))
    log('書きました:', list(files) + ['versions/%s.json' % VERSION, 'reports/s4/cv_result.json'], '例', rows[-1]['key'], {k: round(v, 4) for k, v in out['origin'].items()})


if __name__ == '__main__':
    main()
