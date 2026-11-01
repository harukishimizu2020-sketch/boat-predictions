"""本モデル tolmena13-a の特徴(純 Python。numpy 不要)。analysis_20261020/scripts/build_features.py の card_feats・venue_prior・rhist と
common2.add_derived を同じ計算で写したもの。学習の特徴(float32)と一致することを tests で確かめる。

入力 inp(adapt_* が snapshot から作る、aux_compact と同じ欄名):
  race_id, jcd, hd, rno, card{艇: {grade,F,L,startAvg,winAll,win2All,win3All,winLocal,win2Local,win3Local,motor2,motor3}},
  before{windSpeed,windDirection,waveHeight,weatherDegree,waterDegree,weather}, racers{艇: {tenjiTime,startTenjiTime,startTenjiRank,
  tenjiRank,startSinnyu,tilt,weight,weightAdjust,partsExchange}}, ot{艇: {isshuTime,mawariashiTime,chokusenTime}}, race_class,
  history{vp: {vp_*}, boats: {艇: {r_*・rc_*・rca_*・rv_*}}}(tm13_snap.build が公式結果の蓄積から作る)"""
import math, struct

GRADE = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
NAN = float('nan')
B = range(1, 7)


def f32(x):
    """float32 に丸める(学習の X は float32)"""
    if x is None: return NAN
    x = float(x)
    if x != x: return NAN
    try: return struct.unpack('f', struct.pack('f', x))[0]
    except OverflowError: return math.copysign(float('inf'), x)


def isnan(x): return x != x


def _v(d, k):
    v = d.get(k) if d else None
    return NAN if v is None else v


def card_feats(inp):
    f = {}
    card = inp.get('card') or {}; rac = inp.get('racers') or {}; ot = inp.get('ot') or {}; bef = inp.get('before') or {}
    for b in B:
        c = card.get(b) or card.get(str(b)) or {}; x = rac.get(b) or rac.get(str(b)) or {}; t = ot.get(b) or ot.get(str(b)) or {}
        f['b%d_grade' % b] = GRADE.get(c.get('grade'), NAN)
        for key in ('F', 'L', 'startAvg', 'winAll', 'win2All', 'win3All', 'winLocal', 'win2Local', 'win3Local', 'motor2', 'motor3'):
            v = c.get(key); f['b%d_%s' % (b, key)] = NAN if v is None else v
        if f['b%d_winLocal' % b] == 0: f['b%d_winLocal' % b] = NAN
        f['b%d_tenji' % b] = x.get('tenjiTime') if x.get('tenjiTime') not in (None, 0) else NAN
        f['b%d_exST' % b] = x.get('startTenjiTime') if x.get('startTenjiTime') is not None else NAN
        f['b%d_exRank' % b] = x.get('startTenjiRank') if x.get('startTenjiRank') is not None else NAN
        f['b%d_tenjiRank' % b] = x.get('tenjiRank') if x.get('tenjiRank') is not None else NAN
        f['b%d_exCourse' % b] = x.get('startSinnyu') if x.get('startSinnyu') is not None else NAN
        f['b%d_tilt' % b] = x.get('tilt') if x.get('tilt') is not None else NAN
        f['b%d_weight' % b] = x.get('weight') if x.get('weight') else NAN
        f['b%d_wadj' % b] = x.get('weightAdjust') if x.get('weightAdjust') is not None else NAN
        f['b%d_parts' % b] = (1.0 if x.get('partsExchange') else 0.0) if x else NAN
        f['b%d_isshu' % b] = t.get('isshuTime') if t.get('isshuTime') not in (None, 0) else NAN
        f['b%d_mawari' % b] = t.get('mawariashiTime') if t.get('mawariashiTime') not in (None, 0) else NAN
        f['b%d_chokusen' % b] = t.get('chokusenTime') if t.get('chokusenTime') not in (None, 0) else NAN
    f['wind'] = _v(bef, 'windSpeed') if bef else NAN
    f['windDir'] = _v(bef, 'windDirection') if bef else NAN
    f['wave'] = _v(bef, 'waveHeight') if bef else NAN
    f['temp'] = _v(bef, 'weatherDegree') if bef else NAN
    f['water'] = _v(bef, 'waterDegree') if bef else NAN
    w = bef.get('weather') if bef else None
    f['rain'] = float(w in ('雨', '雪')) if w else NAN
    rc = inp.get('race_class') or ''
    f['cls_ippan'] = float(rc == 'is-ippan') if rc else NAN
    f['has_card'] = float(bool(card)); f['has_before'] = float(bool(rac)); f['has_tenji'] = float(bool(ot))
    return f


def _mean(vals):
    """pandas の float32 列の mean(axis=1) と同じ: float32 のまま順に足し、float32 で割る"""
    v = [x for x in vals if not isnan(x)]
    if not v: return NAN
    s = f32(0.0)
    for x in v: s = f32(s + x)
    return f32(s / len(v))


def _std1(vals):   # pandas std(axis=1): ddof=1、欠損を飛ばす
    v = [x for x in vals if not isnan(x)]
    if len(v) < 2: return NAN
    m = sum(v) / len(v); var = sum((m - x) ** 2 for x in v) / (len(v) - 1)   # pandas: float64 で分散 → float32 → 平方根
    return f32(math.sqrt(f32(var)))


def _min(vals):
    v = [x for x in vals if not isnan(x)]
    return min(v) if v else NAN


def _sub(a, b): return NAN if isnan(a) or isnan(b) else f32(a - b)


def derived(X):
    """common2.add_derived と同じ(X は float32 に丸めた値の dict)"""
    for key in ('exST', 'startAvg', 'tenji', 'isshu', 'mawari', 'chokusen', 'winAll', 'motor2', 'rc_st_mean', 'r_st_mean30'):
        m = _mean([X['b%d_%s' % (b, key)] for b in B])
        for b in B: X['b%d_%s_rel' % (b, key)] = _sub(X['b%d_%s' % (b, key)], m)
    for b in range(2, 7):
        X['b%d_dExST_in' % b] = _sub(X['b%d_exST' % b], X['b%d_exST' % (b - 1)])
        X['b%d_dExST_1' % b] = _sub(X['b%d_exST' % b], X['b1_exST'])
        X['b%d_dExST_min' % b] = _sub(X['b%d_exST' % b], _min([X['b%d_exST' % c] for c in range(1, b)]))
        X['b%d_dStAvg_in' % b] = _sub(X['b%d_startAvg' % b], X['b%d_startAvg' % (b - 1)])
        X['b%d_dRcST_in' % b] = _sub(X['b%d_rc_st_mean' % b], X['b%d_rc_st_mean' % (b - 1)])
        X['b%d_dWin_in' % b] = _sub(X['b%d_winAll' % b], X['b%d_winAll' % (b - 1)])
        X['b%d_dWin_1' % b] = _sub(X['b%d_winAll' % b], X['b1_winAll'])
        X['b%d_dTenji_in' % b] = _sub(X['b%d_tenji' % b], X['b%d_tenji' % (b - 1)])
        X['b%d_dTenji_1' % b] = _sub(X['b%d_tenji' % b], X['b1_tenji'])
    X['b1_win_minus_others'] = _sub(X['b1_winAll'], _mean([X['b%d_winAll' % b] for b in range(2, 7)]))
    X['exST_spread'] = _std1([X['b%d_exST' % b] for b in B])
    X['venue_code'] = float(int(X['_venue']))
    X['is_omura'] = 1.0 if X['_venue'] == '24' else 0.0
    X['exCourse_natural'] = float(sum(1 for b in B if X['b%d_exCourse' % b] == b))
    return X


def features(inp):
    """inp → {列名: float32 の値}(学習の X と同じ列)"""
    raw = {'rno': inp['rno']}
    raw.update(card_feats(inp))
    h = inp.get('history') or {}
    raw.update(h.get('vp') or {})
    hb = h.get('boats') or {}
    for b in B:
        for k, v in (hb.get(b) or hb.get(str(b)) or {}).items(): raw['b%d_%s' % (b, k)] = v
    if inp.get('program'):   # 番組表の特徴(候補 I。列に無い版では使われない)
        import tm13_program as TP
        raw.update(TP.features(inp['program']))
    X = {k: f32(v) for k, v in raw.items()}
    X['_venue'] = str(inp['jcd']).zfill(2)
    for b in B:   # 履歴が無い(新人・蓄積なし)ときは欠け
        for k in ('rc_st_mean', 'r_st_mean30'): X.setdefault('b%d_%s' % (b, k), NAN)
    return derived(X)


def vector(inp, cols):
    X = features(inp)
    return [X.get(c, NAN) for c in cols]
