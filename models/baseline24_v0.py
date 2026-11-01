"""24場の基準モデル baseline24-v0(tools/build_baseline24.py が作成。係数は同じフォルダの baseline24_v0_params.json)。
会場ごとに毎レース同じ確率を出す: 起点7分類・13類・2〜6号艇の動き・コースごとの ST(平均と標準偏差)。
作り方: 2026-08-31 以前の valid かつ判定済みのレース(判定し直しの前のラベル)。判定の少ない会場は全国の割合へ寄せる
(寄せる強さは 〜6/30 で作り 7/1〜8/31 の対数損失で選んだ)。ST は 2025-12-15〜2026-08-31 の会場ごとの実際の ST。
判定し直しの後に作り直す版は baseline24-v1(この版は変えない)。進入は枠なりを予想する。成功確率・1号艇は出さない(null)。"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
_P = None


def _params():
    global _P
    if _P is None:
        with open(os.path.join(HERE, 'baseline24_v0_params.json'), encoding='utf-8') as f: _P = json.load(f)
    return _P


def _norm(d):
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}


def predict(snapshot):
    P = _params()
    jcd = str(snapshot.get('key') or snapshot.get('race_id'))[:2]
    origin = _norm(P['origin'].get(jcd) or P['national']['origin'])
    c13 = _norm(P['class13'].get(jcd) or P['national']['class13'])
    mv = P['move'].get(jcd) or P['national']['move']
    st = P['st'].get(jcd) or P['national']['st']
    boats = []
    for b in range(2, 7):
        m = _norm(mv[str(b)])
        if b == 2: m = {'差し': m['差し'] + m['まくり差し'], 'まくり': m['まくり'], 'まくり差し': 0.0}
        boats.append({'boat': b, 'move_p': m, 'success_p': None})
    return {
        'origin': origin,
        'class13': c13,
        'premise': {'entry': {'courses': [1, 2, 3, 4, 5, 6], 'p_as_predicted': None},
                    'st': [{'boat': b, 'mean': st[str(b)][0], 'sd': st[str(b)][1]} for b in range(1, 7)], 'form': None},
        'scenario_id': None,
        'boats': boats,
        'boat1': None,
        'rationale': [],
    }
