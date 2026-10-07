"""S3 の基準モデル(baseline-v1)と、評価で使う判定ラベルの表(labels/soft-v1.json)を作る。
基準 = 大村の構築期間(2026-07-02〜08-31)の「起点の単純な発生率」(設計書 第6章 S3・GATE 第2章)。毎レース同じ確率を出す。
あわせて、外れ分類を試すために、コースごとの動きの発生率(E3 用)と、コースごとの実際のSTの平均とばらつき(E2 用)も出す。
使い方: python tools/build_baseline.py --results results_omura.jsonl --judge judge_omura.json [--write]
  --write を付けると labels/soft-v1.json・models/baseline_v1.py・versions/baseline-v1.json を新規に書く(既にあれば止まる)。"""
import argparse, collections, json, math, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import predrec as pr
import evaluate as E

OMURA_FROM, SPLIT, OMURA_TO = '20260702', '20260901', '20261001'  # analysis5_origin.py / GATE と同じ期間の分け方
ST_FROM = '20251215'  # STは判定が要らないので、公式結果のある最初の日から構築期間の終わりまで使う

# 設計書5.5 の表と、a5_common.py の SOFT2(2号艇)。値は kyotei_mark1/pipeline/a5_common.py(2026-10-07 修正後)と同じ
TABLE = {'差し|中': {'差し': .92, 'まくり差し': .08}, '差し|低': {'差し': .84, 'まくり差し': .16},
         'まくり差し|中': {'まくり差し': .89, '差し': .08, 'まくり': .03}, 'まくり差し|低': {'まくり差し': .72, '差し': .20, 'まくり': .08},
         'まくり|中': {'まくり': .62, 'まくり差し': .33, '差し': .05}, 'まくり|低': {'まくり': .32, 'まくり差し': .55, '差し': .13}}
TABLE2 = {'差し|中': {'差し': 1.0}, '差し|低': {'差し': .87, 'まくり': .13}}


def races(res, jd, lo, hi):
    """判定済み・枠なり・除外なしの大村のレース(日付・レース番号の順)"""
    out = []
    for k in sorted(jd):
        r = res.get(k)
        if r and lo <= r['hd'] < hi and E.race_valid(r)[0]: out.append((k, r))
    return out


def build(res, jd):
    lab = {'version': 'soft-v1', 'moves': pr.MOVES, 'high_self': 0.97, 'table': TABLE, 'table2': TABLE2,
           'rules': {'winner_kimarite_certain': True, 'boat2_makurizashi_to_sashi': True}, 'fallback': {}}
    con = races(res, jd, OMURA_FROM, SPLIT)
    # 判断不可・欠損の置き換え: 構築期間の同じコースの平均(analysis5_origin.py は会場×コースの全期間平均。ここは評価期間を見ないように構築期間だけ)
    acc = collections.defaultdict(list)
    for k, r in con:
        for b in range(2, 7):
            t, c = E.judge_label(jd[k], b); v = E.soft_move(b, t, c, lab, r)
            if v is not None: acc[b].append(v)
    lab['fallback'] = {str(b): [round(sum(x[i] for x in v) / len(v), 6) for i in range(3)] for b, v in sorted(acc.items())}
    for b, v in lab['fallback'].items():  # 丸めで合計がずれた分を最大の項で直す
        i = max(range(3), key=lambda j: v[j]); v[i] = round(v[i] + 1 - sum(v), 6)
    origin, move = [], collections.defaultdict(list)
    for k, r in con:
        P = {}
        for b in range(2, 7):
            t, c = E.judge_label(jd[k], b); v = E.soft_move(b, t, c, lab, r)
            P[b] = v if v is not None else lab['fallback'][str(b)]; move[b].append(P[b])
        origin.append(E.origin_soft(P))
    base = [sum(o[i] for o in origin) / len(origin) for i in range(len(pr.ORIGIN_CLASSES))]
    mv = {b: [sum(x[i] for x in v) / len(v) for i in range(3)] for b, v in sorted(move.items())}
    st = collections.defaultdict(list)
    for k, r in res.items():
        if str(r['jcd']) == '24' and ST_FROM <= r['hd'] < SPLIT and E.race_valid(r)[0]:
            for b in range(1, 7):
                a = E.actual_st(r, b)
                if a is not None: st[b].append(a)
    stp = {b: (E.mean(v), E.sd(v), len(v)) for b, v in sorted(st.items())}
    return lab, base, mv, stp, len(con)


MODEL = '''"""S3 の基準モデル baseline-v1(tools/build_baseline.py が作成)。毎レース同じ確率を出す。
起点: 大村の構築期間 %(period)s の判定済み・枠なり・除外なし %(n)d レースの起点の割合(ソフトラベル soft-v1)。
動き(2〜6号艇): 同じ期間のコースごとの動きの割合。ST: 大村の %(st_period)s の枠なり・除外なしレースのコースごとの実際のSTの平均と標準偏差。
進入は枠なりを予想する。成功確率・1号艇・足の評価は出さない(null)。"""

ORIGIN = %(origin)s
MOVE = %(move)s
ST = %(st)s


def predict(snapshot):
    return {
        'origin': dict(ORIGIN),
        'premise': {'entry': {'courses': [1, 2, 3, 4, 5, 6], 'p_as_predicted': None},
                    'st': [{'boat': b, 'mean': ST[b][0], 'sd': ST[b][1]} for b in range(1, 7)], 'form': None},
        'scenario_id': None,
        'boats': [{'boat': b, 'move_p': dict(MOVE[b]), 'success_p': None} for b in range(2, 7)],
        'boat1': None,
        'rationale': [],
    }
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', required=True); ap.add_argument('--judge', required=True)
    ap.add_argument('--write', action='store_true'); a = ap.parse_args()
    res = E.load_results(a.results); jd = E.load_judge(a.judge)
    lab, base, mv, stp, n = build(res, jd)
    print('構築期間のレース数', n)
    print('起点の割合', {c: round(x, 4) for c, x in zip(pr.ORIGIN_CLASSES, base)})
    print('動きの割合', {b: [round(x, 4) for x in v] for b, v in mv.items()})
    print('ST(平均, sd, 件数)', {b: (round(m, 4), round(s, 4), k) for b, (m, s, k) in stp.items()})
    print('判断不可の置き換え', lab['fallback'])
    if not a.write: return
    r6 = lambda v: [round(x, 6) for x in v]
    origin = dict(zip(pr.ORIGIN_CLASSES, r6(base)))
    i = max(origin, key=origin.get); origin[i] = round(origin[i] + 1 - sum(origin.values()), 6)
    move = {}
    for b, v in mv.items():
        v = r6(v); j = max(range(3), key=lambda k: v[k]); v[j] = round(v[j] + 1 - sum(v), 6)
        move[b] = dict(zip(pr.MOVES, v))
    files = {
        'labels/soft-v1.json': json.dumps(lab, ensure_ascii=False, indent=1) + '\n',
        'models/baseline_v1.py': MODEL % {'period': '%s〜%s' % (OMURA_FROM, '20260831'), 'n': n, 'st_period': '%s〜%s' % (ST_FROM, '20260831'),
                                          'origin': json.dumps(origin, ensure_ascii=False), 'move': repr(move),
                                          'st': repr({b: (round(m, 4), round(s, 4)) for b, (m, s, _) in stp.items()})},
    }
    for p, txt in files.items():
        fp = os.path.join(HERE, p)
        if os.path.exists(fp): sys.exit('既にあります(上書きしない): ' + p)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, 'w', encoding='utf-8', newline='\n') as f: f.write(txt)
    man = {'version': 'baseline-v1', 'purpose': 'S3 の基準モデル(評価器の検証と、本番の版の比べる相手)。この版自体は段階2→3の判定に使わない',
           'model': '大村の構築期間の起点の単純な発生率(全レース同じ確率)。動き・STもコースごとの単純な割合・平均',
           'frozen_at': pr.now_utc(), 'built_from': {'data': 'kyotei_mark1/data/results_all.jsonl・data/judge/[ab]_*.json(後のファイル優先)',
                                                     'period': '%s〜20260831(起点・動き %d レース)、ST は %s〜20260831' % (OMURA_FROM, n, ST_FROM),
                                                     'labels': 'labels/soft-v1.json'},
           'eval_from': None, 'origin_classes': pr.ORIGIN_CLASSES, 'bands': {'lo': 0.0, 'hi': 1.01},
           'bands_method': '全レース同じ確率のため帯を分けない(全レース○)', 'success_definition': None, 'escape_definition': None,
           'soft_label_table': 'labels/soft-v1.json', 'entry': 'models/baseline_v1.py', 'code_files': ['models/baseline_v1.py'],
           'code_files_hash': pr.code_files_hash(['models/baseline_v1.py'], HERE),
           'notes': ['評価期間(2026-09-01〜10-01)は作るときに見ていない(設計書5.3)', '成功確率・1号艇・足の評価は出さない(null)']}
    fp = os.path.join(HERE, 'versions', 'baseline-v1.json')
    if os.path.exists(fp): sys.exit('既にあります(上書きしない): versions/baseline-v1.json')
    with open(fp, 'w', encoding='utf-8', newline='\n') as f: json.dump(man, f, ensure_ascii=False, indent=1); f.write('\n')
    pr.check_manifest(man); print('書きました:', list(files) + ['versions/baseline-v1.json'])


if __name__ == '__main__':
    main()
