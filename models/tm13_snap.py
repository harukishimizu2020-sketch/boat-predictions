"""本モデル tolmena13-a の snapshot を補う関数 build(snap, ctx)(純 Python)。
公式結果の蓄積から、そのレースの日より前の日のデータだけで履歴の特徴を作り、snap['tm13_history'] に入れて返す。
計算は build_features.py の venue_prior・rhist と同じ(会場×365日の1着コース・決まり手、選手の直近30走・365日・コース別・会場別)。
ctx: {'results': 公式結果の行の iterable か、jsonl(.gz)のパス, 'date': 'YYYYMMDD'(省けば snap のレースの日)}。
結果の行は races_core と同じ形(key, jcd, hd, rno, order, kimarite, course, st, tobans, excluded, valid)。valid が無ければ
「除外なし・枠なり・着順6つ・登番6つ」で判定する(evaluate.race_valid と build_core と同じ)。"""
import bisect, datetime, gzip, json, math

KMI = {'まくり': 1, '差し': 2, 'まくり差し': 3, '逃げ': 4}
_D0 = datetime.date(2020, 1, 1)
NAN = float('nan')


def dnum(d): return (datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])) - _D0).days


def st_val(s):
    if not s: return NAN, 0.0
    s = str(s)
    if s.startswith('F'): return NAN, 1.0
    if s.startswith('L'): return NAN, 0.0
    try: return float(s), 0.0
    except ValueError: return NAN, 0.0


def is_valid(r):
    if 'valid' in r: return bool(r['valid'])
    order = r.get('order') or []; tob = r.get('tobans') or {}
    return (not r.get('excluded')) and list(r.get('course') or []) == [1, 2, 3, 4, 5, 6] and len(order) == 6 and len(tob) == 6


def _rows(src):
    if isinstance(src, str):
        op = gzip.open if src.endswith('.gz') else open
        with op(src, 'rt', encoding='utf-8') as f:
            for l in f:
                if l.strip(): yield json.loads(l)
    else:
        yield from src


class ResultsIndex:
    """公式結果の索引: 選手ごとの走(新しい順)と会場ごとの (日, 1着の艇, 決まり手)"""
    def __init__(self, src):
        runs = {}; ven = {}
        for r in _rows(src):
            if not is_valid(r): continue
            order = [int(x) for x in r['order']]; dn = dnum(r['hd']); jcd = str(r['jcd']).zfill(2)
            ven.setdefault(jcd, []).append((dn, order[0], KMI.get(r.get('kimarite'), 0)))
            for b in range(1, 7):
                t = (r.get('tobans') or {}).get(str(b))
                if not t: continue
                fin = order.index(b) + 1 if b in order else 0
                st, F = st_val((r.get('st') or {}).get(str(b)))
                km = KMI.get(r.get('kimarite'), 0) if fin == 1 else 0
                runs.setdefault(str(t), []).append((dn, int(r['rno']), int(jcd), b, fin, st, F, km))
        for t in runs: runs[t].sort(key=lambda x: (x[0], x[1]), reverse=True)
        for j in ven: ven[j].sort()
        self.runs = runs; self.ven = ven; self.vdays = {j: [x[0] for x in L] for j, L in ven.items()}

    def venue_prior(self, jcd, date):
        dn = dnum(date); L = self.ven.get(jcd, []); d = self.vdays.get(jcd, [])
        S = L[bisect.bisect_left(d, dn - 365):bisect.bisect_left(d, dn)]; n = len(S); f = {'vp_n': n}
        for c in range(1, 7): f['vp_win%d' % c] = (sum(1 for x in S if x[1] == c) + 1) / (n + 6)
        for i, nm in ((1, 'makuri'), (2, 'sashi'), (3, 'mz'), (4, 'nige')): f['vp_km_' + nm] = (sum(1 for x in S if x[2] == i) + 1) / (n + 5)
        return f

    def racer(self, toban, date, course, venue):
        dn = dnum(date); A = [x for x in self.runs.get(str(toban), []) if x[0] < dn]
        rec = [x for x in A if x[0] >= dn - 365]; f = {}
        f['r_n365'] = len(rec); f['r_nall'] = len(A); f['r_days_since'] = dn - A[0][0] if A else NAN
        a30 = A[:30]; st = [x[5] for x in a30 if x[5] == x[5]]
        f['r_st_mean30'] = _mean(st); f['r_st_sd30'] = _std0(st) if len(st) >= 3 else NAN
        f['r_F365'] = float(sum(x[6] for x in rec))
        fin = [x[4] for x in a30 if x[4] > 0]
        f['r_fin_mean30'] = _mean(fin); f['r_win30'] = _mean([float(v == 1) for v in fin]); f['r_top2_30'] = _mean([float(v <= 2) for v in fin])
        f10 = [x[4] for x in A[:10] if x[4] > 0]; f['r_fin_mean10'] = _mean(f10)
        cc = [x for x in rec if x[3] == course]; fc = [x[4] for x in cc if x[4] > 0]
        f['rc_n'] = len(cc)
        f['rc_win'] = (sum(1 for v in fc if v == 1) + 2 / 6) / (len(fc) + 2)
        f['rc_top2'] = (sum(1 for v in fc if v <= 2) + 4 / 6) / (len(fc) + 2)
        f['rc_top3'] = (sum(1 for v in fc if v <= 3) + 1.0) / (len(fc) + 2)
        f['rc_fin_mean'] = _mean(fc)
        stc = [x[5] for x in cc if x[5] == x[5]]; f['rc_st_mean'] = _mean(stc); f['rc_st_sd'] = _std0(stc) if len(stc) >= 3 else NAN
        f['rc_F'] = float(sum(x[6] for x in cc))
        for i, nm in ((1, 'makuri'), (2, 'sashi'), (3, 'mz'), (4, 'nige')): f['rc_win_' + nm] = sum(1 for x in cc if x[7] == i) / (len(cc) + 2)
        ca = [x for x in A if x[3] == course]; fa = [x[4] for x in ca if x[4] > 0]
        f['rca_n'] = len(ca); f['rca_win'] = (sum(1 for v in fa if v == 1) + 2 / 6) / (len(fa) + 2)
        lv = [x for x in A if x[2] == int(venue) and x[0] >= dn - 730]; fv = [x[4] for x in lv if x[4] > 0]
        f['rv_n'] = len(lv); f['rv_fin_mean'] = _mean(fv); f['rv_win'] = _mean([float(v == 1) for v in fv])
        return f


def _mean(v): return sum(v) / len(v) if v else NAN
def _std0(v):
    m = sum(v) / len(v); return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def history(idx, jcd, hd, tobans):
    """{'vp': {...}, 'boats': {'1': {...}, ...}}。NaN は JSON に書けないので None にする"""
    h = {'vp': idx.venue_prior(jcd, hd), 'boats': {}}
    for b in range(1, 7):
        t = tobans.get(str(b)) if tobans else None
        h['boats'][str(b)] = idx.racer(t, hd, b, jcd) if t else idx.racer('-', hd, b, jcd)
    return _nan_to_none(h)


def _nan_to_none(o):
    if isinstance(o, dict): return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, float) and o != o: return None
    return o


_IDX = {}


def build(snap, ctx):
    """snap(daily_live の snapshot)に tm13_history を足して返す。登番・日付・会場は tm13_adapt.race_info で取る"""
    import tm13_adapt as AD
    info = AD.race_info(snap)
    src = ctx['results']
    if isinstance(src, str): key = src
    else:   # 行の並び(daily_live は毎回新しいリストを渡す)→ 件数と最初・最後の key で使い回す
        src = list(src); key = (len(src), src[0].get('key') if src else None, src[-1].get('key') if src else None)
    if key not in _IDX:
        _IDX.clear(); _IDX[key] = ResultsIndex(src)
    out = dict(snap); out['tm13_history'] = history(_IDX[key], info['jcd'], ctx.get('date') or info['hd'], info['tobans'])
    out['tm13_history']['built_from'] = {'date': ctx.get('date') or info['hd'], 'rule': 'その日より前の日の valid レースだけ'}
    return out
