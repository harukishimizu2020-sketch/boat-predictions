"""本モデル tolmena13-a(予測精度スレッド 2026-11-01)。predict(snapshot) → prediction_record の出力の形。
中身: 2段+2着(1着×決まり手17類と2着を GBM、学習側の「その条件で起きた展開の割合」の表を掛け合わせる)。木の数 600・学習率0.02・葉15。
13類の確率・起点7分類・2〜6号艇の動きを同じ混ぜ方で出す。純 Python(numpy・lightgbm 不要)。同じ入力なら同じ出力。
係数は同じフォルダの tm13_a_params.json.gz(train_a.py が作る。環境変数では差し替えない)。snapshot は tm13_snap.build で履歴を足したもの。
自信度は predrec の conf-v0(13類の上位3つの合計=主、最大=副)で、モデルは書かない。"""
import gzip, json, os
import tm13_features as F, tm13_trees as TT, tm13_adapt as AD

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, 'tm13_a_params.json.gz')   # 版のハッシュに入る係数だけを読む(環境変数で差し替えない。試験は load_params で別のものを読む)
_P = None
MOVES = ['差し', 'まくり', 'まくり差し']


def params():
    global _P
    if _P is None:
        with gzip.open(PARAMS, 'rt', encoding='utf-8') as f: _P = json.load(f)
    return _P


def load_params(path):
    """試験用: 別の係数を読む(本番の記録では使わない)"""
    global _P
    with gzip.open(path, 'rt', encoding='utf-8') as f: _P = json.load(f)
    return _P


def _mix(P, W, PS, T, T2, clip):
    K = len(T[0]); out = [0.0] * K
    for j, name in enumerate(P['wk']):
        if name == '逆転':
            for k in range(K): out[k] += W[j] * T[j][k]
            continue
        w = int(name[0])
        for s in range(1, 7):
            if s == w: continue
            c = W[j] * PS[w][s - 1]; row = T2[j][s]
            for k in range(K): out[k] += c * (max(row[k], 1e-9) if clip else row[k])
    return out


def _norm(v):
    z = sum(v); return [x / z for x in v]


def raw_predict(inp):
    """(27類の確率, 起点と動きの表の値, 特徴ベクトル)"""
    P = params(); x = F.vector(inp, P['cols'])
    W = TT.predict_proba(P['win'], x)
    PS = {}
    for w in range(1, 7):
        q = TT.predict_proba(P['second'], x + [float(w)]); q[w - 1] = 0.0; PS[w] = _norm(q)
    p27 = _norm(_mix(P, W, PS, P['T'], P['T2'], True))
    p27 = _norm([max(v, 1e-9) for v in p27])
    z = _mix(P, W, PS, P['TZ'], P['TZ2'], False)
    return p27, z, x, W


def predict(snapshot):
    P = params(); inp = AD.to_inp(snapshot)
    p27, z, x, W = raw_predict(inp)
    c13 = {c: 0.0 for c in P['c13']}
    for i, j in enumerate(P['m13']): c13[P['c13'][j]] += p27[i]
    origin = dict(zip(P['origin_classes'], _norm(z[:7])))
    boats = []
    for b in range(2, 7):
        m = _norm(z[7 + 3 * (b - 2): 10 + 3 * (b - 2)])
        mp = dict(zip(MOVES, m))
        if b == 2: mp = {'差し': mp['差し'] + mp['まくり差し'], 'まくり': mp['まくり'], 'まくり差し': 0.0}
        boats.append({'boat': b, 'move_p': mp, 'success_p': None})
    hb = (inp.get('history') or {}).get('boats') or {}
    st = []
    for b in range(1, 7):
        h = hb.get(str(b)) or {}; mu, sd = P['st_course'][str(b)]
        m_ = h.get('rc_st_mean') if h.get('rc_st_mean') is not None else h.get('r_st_mean30')
        s_ = h.get('rc_st_sd') if h.get('rc_st_sd') is not None else h.get('r_st_sd30')
        st.append({'boat': b, 'mean': round(m_ if m_ is not None else mu, 4), 'sd': round(max(s_ if s_ is not None else sd, 0.02), 4)})
    wk = dict(zip(P['wk'], W))
    return {
        'origin': origin, 'class13': c13,
        'premise': {'entry': {'courses': [1, 2, 3, 4, 5, 6], 'p_as_predicted': None}, 'st': st, 'form': None},
        'scenario_id': None, 'boats': boats, 'boat1': None,
        'rationale': [{'boat': 1, 'feature': '1号艇が逃げる確率', 'value': round(wk.get('1逃げ', 0.0), 4), 'target': '1逃げ',
                       'direction': 'up' if wk.get('1逃げ', 0.0) >= 0.5 else 'down', 'effect': None}],
    }
