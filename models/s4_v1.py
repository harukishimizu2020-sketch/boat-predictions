"""S4 本モデル s4-v1(tools/build_s4.py が学習・凍結。係数は同じフォルダの s4_v1_params.json)。

入力 snapshot(レース前に分かるものだけ。tools/backtest_s4.py・build_s4.py の make_snapshot が作る):
  race_id, jcd, hd, rno, tobans, card(出走表), before(直前情報: 風・波・展示), original_tenjis(オリジナル展示),
  race_class, history(艇ごとの、その日より前の判定済みレースからの選手×コースの動きの合計と、過去20走のST)。
  オッズと本番STは入れない(設計メモ 2.2)。
出力は baseline_v2.py と同じ形(origin・premise(entry・st)・boats の move_p など)。

計算(pure Python。numpy の無い端末でも動く):
  1) features(snapshot, fp): 特徴ベクトル(欠損は None)。学習(build_s4.py)もこの同じ関数を使う。
  2) 欠損は学習データの中央値で埋め、学習データの平均・標準偏差で標準化する。
  3) 起点: 候補 A = 起点7分類の多項ロジスティック、候補 B = 2〜6号艇の動き3分類のロジスティックを origin_soft で合成。
     どちらを使うかと混ぜる割合 λ は params の choice(交差検証で決めて凍結)。 p = (1-λ)·基準(baseline-v2) + λ·モデル。
  4) 動き(2〜6号艇): 動き3分類のロジスティックを基準の動きの割合と λ で混ぜる。2号艇のまくり差しは 0。
  5) ST: コースごとの切片 + β×(選手の過去20走の平均ST − 中心)。sd は構築期間の残差の標準偏差(過去走が無い艇は素の sd)。
同じ入力なら同じ出力を返す(浮動小数の演算順を固定している)。"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(HERE, 's4_v1_params.json')
ORIGIN_CLASSES = ['2まくり', '3まくり', '4まくり', '5まくり', '6まくり', '3まくり差し', 'その他']
MOVES = ['差し', 'まくり', 'まくり差し']
_PARAMS = None


def _num(v):
    """数値に直す。None・空・文字(F/L付き)・NaN は None。本番STの形(F.05)はここでは使わない(入力に入れない)"""
    if v is None or isinstance(v, bool): return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float('inf'), float('-inf')): return None
    return x


def _sub(a, b):
    return None if a is None or b is None else a - b


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _min(xs):
    xs = [x for x in xs if x is not None]
    return min(xs) if xs else None


def _shrink(s, n, prior, k):
    """履歴の割合を事前の割合へ縮める: (合計 + k×事前) / (件数 + k)"""
    return (s + k * prior) / (n + k)


def features(snapshot, fp):
    """(名前の一覧, 値の一覧)。fp = params['feature_params'](履歴の事前割合・縮める強さ・会場の一覧)。
    値は float か None(欠損)。順番は固定で、学習時の params['feature_names'] と一致していなければならない"""
    card = {int(c.get('boat')): c for c in (snapshot.get('card') or []) if c and c.get('boat') is not None}
    bef = snapshot.get('before') or {}
    racers = {int(r.get('boatNumber')): r for r in (bef.get('racers') or []) if r and r.get('boatNumber') is not None}
    ot = {int(o.get('boatNumber')): o for o in (snapshot.get('original_tenjis') or []) if o and o.get('boatNumber') is not None}
    H = snapshot.get('history') or {}
    k = float(fp['hist_k'])
    names, vals = [], []

    def add(name, v):
        names.append(name); vals.append(None if v is None else float(v))

    win = {b: _num(card.get(b, {}).get('winAll')) for b in range(1, 7)}
    win2 = {b: _num(card.get(b, {}).get('win2All')) for b in range(1, 7)}
    avg = {b: _num(card.get(b, {}).get('startAvg')) for b in range(1, 7)}
    fcnt = {b: _num(card.get(b, {}).get('F')) for b in range(1, 7)}
    motor = {b: _num(card.get(b, {}).get('motor2')) for b in range(1, 7)}
    grade = {b: card.get(b, {}).get('grade') for b in range(1, 7)}
    exst = {b: _num(racers.get(b, {}).get('startTenjiTime')) for b in range(1, 7)}
    tenji = {b: _num(racers.get(b, {}).get('tenjiTime')) for b in range(1, 7)}
    tilt = {b: _num(racers.get(b, {}).get('tilt')) for b in range(1, 7)}
    weight = {b: _num(racers.get(b, {}).get('weight')) for b in range(1, 7)}
    isshu = {b: _num(ot.get(b, {}).get('isshuTime')) for b in range(1, 7)}
    mawari = {b: _num(ot.get(b, {}).get('mawariashiTime')) for b in range(1, 7)}
    p20 = {}
    for b in range(1, 7):
        h = (H.get(str(b)) or {}).get('st') or {}
        n = _num(h.get('n')) or 0.0
        p20[b] = _num(h.get('mean')) if n >= 5 else None
    tenji_m, isshu_m, mawari_m = _mean(tenji.values()), _mean(isshu.values()), _mean(mawari.values())
    for b in range(1, 7):
        add('win_%d' % b, win[b]); add('win2_%d' % b, win2[b]); add('avgst_%d' % b, avg[b])
        add('a1_%d' % b, None if grade[b] is None else float(grade[b] == 'A1'))
        add('a2_%d' % b, None if grade[b] is None else float(grade[b] == 'A2'))
        add('f_%d' % b, fcnt[b]); add('motor_%d' % b, motor[b])
        add('exst_%d' % b, exst[b]); add('tenji_rel_%d' % b, _sub(tenji[b], tenji_m))
        add('tilt_%d' % b, tilt[b]); add('weight_%d' % b, weight[b])
        add('isshu_rel_%d' % b, _sub(isshu[b], isshu_m)); add('mawari_rel_%d' % b, _sub(mawari[b], mawari_m))
        add('p20_%d' % b, p20[b])
        if b >= 2:
            inner_ex = [exst[j] for j in range(1, b)]
            add('exst_d1_%d' % b, _sub(exst[b], exst[b - 1])); add('exst_dmin_%d' % b, _sub(exst[b], _min(inner_ex)))
            add('avgst_d1_%d' % b, _sub(avg[b], avg[b - 1])); add('p20_d1_%d' % b, _sub(p20[b], p20[b - 1]))
            add('p20_dmin_%d' % b, _sub(p20[b], _min([p20[j] for j in range(1, b)])))
            add('win_d1_%d' % b, _sub(win[b], win[b - 1]))
            h = H.get(str(b)) or {}
            sc = h.get('same_course') or {}; ac = h.get('any_course') or {}
            n_s = _num(sc.get('n')) or 0.0; n_a = _num(ac.get('n')) or 0.0
            pr = fp['hist_prior'][str(b)]; pa = fp['hist_prior_any']
            add('hist_makuri_%d' % b, _shrink(_num(sc.get('まくり')) or 0.0, n_s, pr[1], k))
            add('hist_makurizashi_%d' % b, _shrink(_num(sc.get('まくり差し')) or 0.0, n_s, pr[2], k))
            add('hist_n_%d' % b, math.log1p(n_s))
            add('histany_makuri_%d' % b, _shrink(_num(ac.get('まくり')) or 0.0, n_a, pa[1], k))
            add('histany_makurizashi_%d' % b, _shrink(_num(ac.get('まくり差し')) or 0.0, n_a, pa[2], k))
            add('histany_n_%d' % b, math.log1p(n_a))
    ws = _num(bef.get('windSpeed')); wd = _num(bef.get('windDirection')); wave = _num(bef.get('waveHeight'))
    add('wind_speed', ws); add('wave', wave)
    if wd is not None and 1 <= wd <= 16:
        ang = (wd - 1) / 16.0 * 2 * math.pi
        add('wind_sin', math.sin(ang) * (ws if ws is not None else 0.0)); add('wind_cos', math.cos(ang) * (ws if ws is not None else 0.0))
    else:
        add('wind_sin', 0.0 if wd is not None else None); add('wind_cos', 0.0 if wd is not None else None)
    w = bef.get('weather')
    add('rain', None if w is None else float(w == '雨')); add('sunny', None if w is None else float(w == '晴'))
    add('temp', _num(bef.get('weatherDegree'))); add('water_temp', _num(bef.get('waterDegree')))
    add('rno', _num(snapshot.get('rno')))
    rc = snapshot.get('race_class')
    add('ippan', None if rc is None else float(rc == 'is-ippan'))
    jcd = str(snapshot.get('jcd') or '')
    for v in fp['venues']:
        add('venue_%s' % v, float(jcd == v))
    return names, vals


# ---------- 線形モデルの計算(pure Python) ----------
def _standardize(vals, params):
    med, mu, sd = params['impute_median'], params['scale_mean'], params['scale_sd']
    out = []
    for i, v in enumerate(vals):
        x = med[i] if v is None else v
        s = sd[i]
        out.append((x - mu[i]) / s if s > 0 else 0.0)
    return out


def _softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]


def _linear(x, model):
    """model: {'coef': [[...] × クラス数], 'intercept': [...]} → 各クラスの確率"""
    z = []
    for c in range(len(model['intercept'])):
        w = model['coef'][c]
        acc = model['intercept'][c]
        for i in range(len(x)):
            acc += w[i] * x[i]
        z.append(acc)
    return _softmax(z)


def origin_soft(P):
    """evaluate.origin_soft と同じ計算(pure Python)。P: {艇番: [差し, まくり, まくり差し]}"""
    out = {}; none_m = 1.0
    for b in range(2, 7):
        pm = P[b][1]; out['%dまくり' % b] = out.get('%dまくり' % b, 0.0) + none_m * pm; none_m *= (1 - pm)
    rest = none_m; none_ms = 1.0
    for b in range(2, 7):
        q = P[b][2] / max(1e-9, 1 - P[b][1])
        out['%dまくり差し' % b] = out.get('%dまくり差し' % b, 0.0) + rest * none_ms * q; none_ms *= (1 - q)
    out['全艇差し'] = rest * none_ms
    v = [0.0] * len(ORIGIN_CLASSES)
    for kk, p in out.items():
        v[ORIGIN_CLASSES.index(kk) if kk in ORIGIN_CLASSES else ORIGIN_CLASSES.index('その他')] += p
    s = sum(v)
    return [x / s for x in v]


def _mix(base, model, lam):
    return [(1 - lam) * a + lam * b for a, b in zip(base, model)]


def load_params():
    global _PARAMS
    if _PARAMS is None:
        with open(PARAMS_FILE, encoding='utf-8') as f:
            _PARAMS = json.load(f)
    return _PARAMS


def compute(snapshot, params):
    """予測の中身(起点・動き・ST)を返す。predict はこれを記録の形に包む"""
    names, vals = features(snapshot, params['feature_params'])
    if names != params['feature_names']: raise ValueError('特徴の並びが学習時と違います')
    x = _standardize(vals, params)
    ch = params['choice']
    base_o = [params['base_origin'][c] for c in ORIGIN_CLASSES]
    base_m = {b: [params['base_move'][str(b)][m] for m in MOVES] for b in range(2, 7)}
    # 動き(候補 B の部品)。2号艇は [差し, まくり] の2分類
    M = {}
    for b in range(2, 7):
        p = _linear(x, params['move_models'][str(b)])
        M[b] = [p[0], p[1], 0.0] if b == 2 else p
    lam_m = ch['move_lambda']
    moves = {b: _mix(base_m[b], M[b], lam_m) for b in range(2, 7)}
    if ch['candidate'] == 'A':
        model_o = _linear(x, params['origin_model'])
    elif ch['candidate'] == 'B':
        model_o = origin_soft(M)
    else:
        model_o = base_o
    lam_o = ch['origin_lambda']
    origin = _mix(base_o, model_o, lam_o)
    s = sum(origin); origin = [v / s for v in origin]
    st = []
    stp = params['st']
    for b in range(1, 7):
        h = ((snapshot.get('history') or {}).get(str(b)) or {}).get('st') or {}
        n = _num(h.get('n')) or 0.0; mean = _num(h.get('mean'))
        if mean is not None and n >= stp['min_n']:
            st.append({'boat': b, 'mean': stp['alpha'][str(b)] + stp['beta'] * (mean - stp['center']), 'sd': stp['sd'][str(b)]})
        else:
            st.append({'boat': b, 'mean': stp['alpha'][str(b)], 'sd': stp['sd0'][str(b)]})
    return {'origin': origin, 'moves': moves, 'st': st, 'names': names, 'vals': vals, 'model_origin': model_o}


def _boat_of(name):
    t = name.rsplit('_', 1)[-1]
    return int(t) if t.isdigit() and 1 <= int(t) <= 6 else None


def rationale_for(r, params, top=8):
    """根拠(記録の rationale の形): 主シナリオの起点のロジットへの寄与が大きい特徴を上から並べる(候補 A のとき)。
    effect は確率への寄与の近似(ポイント、p(1-p)×寄与)。候補 B・基準のときは混ぜ方だけを書く"""
    ch = params['choice']
    out = [{'boat': None, 'feature': '混ぜ方', 'value': '候補 %s、起点 λ=%.2f、動き λ=%.2f' % (ch['candidate'], ch['origin_lambda'], ch['move_lambda']),
            'target': '起点', 'direction': 'up', 'effect': None}]
    if ch['candidate'] != 'A': return out
    x = _standardize(r['vals'], params)
    c = max(range(len(ORIGIN_CLASSES)), key=lambda k: (r['origin'][k], -k))
    w = params['origin_model']['coef'][c]; pc = r['origin'][c]
    contrib = [(w[i] * x[i], i) for i in range(len(x))]
    contrib.sort(key=lambda t: (-abs(t[0]), t[1]))
    for v, i in contrib[:top]:
        if v == 0: break
        out.append({'boat': _boat_of(r['names'][i]), 'feature': r['names'][i], 'value': r['vals'][i], 'target': ORIGIN_CLASSES[c],
                    'direction': 'up' if v > 0 else 'down', 'effect': round(100 * pc * (1 - pc) * v * ch['origin_lambda'], 4)})
    return out


def predict(snapshot):
    params = load_params()
    r = compute(snapshot, params)
    return {
        'origin': dict(zip(ORIGIN_CLASSES, r['origin'])),
        'premise': {'entry': {'courses': [1, 2, 3, 4, 5, 6], 'p_as_predicted': None}, 'st': r['st'], 'form': None},
        'scenario_id': None,
        'boats': [{'boat': b, 'move_p': dict(zip(MOVES, r['moves'][b])), 'success_p': None} for b in range(2, 7)],
        'boat1': None,
        'rationale': rationale_for(r, params),
    }
