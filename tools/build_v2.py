"""判定ラベルの表 soft-v2(大村専用)と、それで作り直した基準モデル baseline-v2 を作る(2026-10-08、S4 と段階3の合格条件の計算し直し用)。
soft-v2 の元: kyotei_mark1/data/mj40_fixed/analysis/mj40_soft_omura.json(手動判定40レース+統一ルール後、大村 301 艇・他会場 377 艇の照合、他会場を重み4の事前分布)。
  - 3〜6号艇は「3-6|ラベル|確信度」の行、2号艇は「2|ラベル|確信度」の行(まくり差しは無い)。確信度「高」の行も表から使う(soft-v1 は 0.97 固定)。
  - 1着艇の決まり手で確定する規則・2号艇の規則は soft-v1 と同じ。1号艇の行(攻められ方)は評価器が使わないので入れない。
  - 判断不可・欠損は soft-v1 と同じく、構築期間の同じコースの平均で置き換える(表の「判断不可」行は使わない。件数が 2〜6 艇と少ないため)。
基準モデル baseline-v2 は build_baseline.py と同じ作り方(構築期間 2026-07-02〜08-31 の起点・動きの割合と、コースごとのST)で、表だけ soft-v2 にしたもの。
使い方: python tools/build_v2.py --results results_omura.jsonl --judge judge_omura.json --mj40 mj40_soft_omura.json [--write]"""
import argparse, collections, hashlib, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tools'))
import predrec as pr
import evaluate as E
import build_baseline as B

CONFS = ('高', '中', '低')


def norm(d, keys):
    v = {m: max(0.0, float(d.get(m, 0.0))) for m in keys}
    s = sum(v.values()); v = {m: round(x / s, 4) for m, x in v.items()}
    i = max(v, key=v.get); v[i] = round(v[i] + 1 - sum(v.values()), 4)
    return {m: x for m, x in v.items() if x > 0}


def make_lab(src, src_md5):
    t = src['table']
    table = {'%s|%s' % (m, c): norm(t['3-6|%s|%s' % (m, c)], pr.MOVES) for m in pr.MOVES for c in CONFS}
    table2 = {'%s|%s' % (m, c): norm(t['2|%s|%s' % (m, c)], ('差し', 'まくり')) for m in ('差し', 'まくり') for c in CONFS}
    return {'version': 'soft-v2', 'moves': pr.MOVES, 'high_self': 0.97, 'table': table, 'table2': table2,
            'rules': {'winner_kimarite_certain': True, 'boat2_makurizashi_to_sashi': True}, 'fallback': {},
            'source': {'file': 'kyotei_mark1/data/mj40_fixed/analysis/mj40_soft_omura.json', 'md5': src_md5,
                       'weight': src.get('weight'), 'omura_matches': src.get('omura_matches'), 'other_matches': src.get('other_matches'),
                       'note': '手動判定40レース(統一ルール: 迷う3・5・6号艇はまくり差し)と判定AIの照合。2号艇のまくり差し行は無い(table2 で差し・まくりのみ)。high_self は表に高の行があるため使われない'}}


def build(res, jd, lab):
    con = B.races(res, jd, B.OMURA_FROM, B.SPLIT)
    acc = collections.defaultdict(list)
    for k, r in con:
        for b in range(2, 7):
            t, c = E.judge_label(jd[k], b); v = E.soft_move(b, t, c, lab, r)
            if v is not None: acc[b].append(v)
    lab['fallback'] = {str(b): [round(sum(x[i] for x in v) / len(v), 6) for i in range(3)] for b, v in sorted(acc.items())}
    for b, v in lab['fallback'].items():
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
        if str(r['jcd']) == '24' and B.ST_FROM <= r['hd'] < B.SPLIT and E.race_valid(r)[0]:
            for b in range(1, 7):
                a = E.actual_st(r, b)
                if a is not None: st[b].append(a)
    stp = {b: (E.mean(v), E.sd(v), len(v)) for b, v in sorted(st.items())}
    return base, mv, stp, len(con)


def main():
    ap = argparse.ArgumentParser()
    for k in ('--results', '--judge', '--mj40'): ap.add_argument(k, required=True)
    ap.add_argument('--write', action='store_true'); a = ap.parse_args()
    raw = open(a.mj40, 'rb').read(); src = json.loads(raw.decode('utf-8'))
    lab = make_lab(src, hashlib.md5(raw).hexdigest())
    res = E.load_results(a.results); jd = E.load_judge(a.judge)
    base, mv, stp, n = build(res, jd, lab)
    print('構築期間のレース数', n)
    print('起点の割合', {c: round(x, 4) for c, x in zip(pr.ORIGIN_CLASSES, base)})
    print('動きの割合', {b: [round(x, 4) for x in v] for b, v in mv.items()})
    print('判断不可の置き換え', lab['fallback'])
    if not a.write: return
    r6 = lambda v: [round(x, 6) for x in v]
    origin = dict(zip(pr.ORIGIN_CLASSES, r6(base)))
    i = max(origin, key=origin.get); origin[i] = round(origin[i] + 1 - sum(origin.values()), 6)
    move = {}
    for b, v in mv.items():
        v = r6(v); j = max(range(3), key=lambda k: v[k]); v[j] = round(v[j] + 1 - sum(v), 6)
        move[b] = dict(zip(pr.MOVES, v))
    model = B.MODEL.replace('S3 の基準モデル baseline-v1(tools/build_baseline.py が作成)', 'S4 と比べる基準モデル baseline-v2(tools/build_v2.py が作成。baseline-v1 と同じ作り方で、表を soft-v2 にしたもの)').replace('ソフトラベル soft-v1', 'ソフトラベル soft-v2')
    files = {'labels/soft-v2.json': json.dumps(lab, ensure_ascii=False, indent=1) + '\n',
             'models/baseline_v2.py': model % {'period': '%s〜%s' % (B.OMURA_FROM, '20260831'), 'n': n, 'st_period': '%s〜%s' % (B.ST_FROM, '20260831'),
                                               'origin': json.dumps(origin, ensure_ascii=False), 'move': repr(move),
                                               'st': repr({b: (round(m, 4), round(s, 4)) for b, (m, s, _) in stp.items()})}}
    for p in list(files) + ['versions/baseline-v2.json']:
        if os.path.exists(os.path.join(HERE, p)): sys.exit('既にあります(上書きしない): ' + p)
    for p, txt in files.items():
        with open(os.path.join(HERE, p), 'w', encoding='utf-8', newline='\n') as f: f.write(txt)
    man = {'version': 'baseline-v2', 'purpose': 'S4 の本モデルと比べる基準モデル(soft-v2 の表で作り直したもの)。この版自体は段階2→3の判定に使わない',
           'model': '大村の構築期間の起点の単純な発生率(全レース同じ確率)。動き・STもコースごとの単純な割合・平均',
           'frozen_at': pr.now_utc(), 'built_from': {'data': 'kyotei_mark1/data/results_all.jsonl・data/judge/[ab]_*.json(後のファイル優先)',
                                                     'period': '%s〜20260831(起点・動き %d レース)、ST は %s〜20260831' % (B.OMURA_FROM, n, B.ST_FROM),
                                                     'labels': 'labels/soft-v2.json'},
           'eval_from': None, 'origin_classes': pr.ORIGIN_CLASSES, 'bands': {'lo': 0.0, 'hi': 1.01},
           'bands_method': '全レース同じ確率のため帯を分けない(全レース○)', 'success_definition': None, 'escape_definition': None,
           'soft_label_table': 'labels/soft-v2.json', 'entry': 'models/baseline_v2.py', 'code_files': ['models/baseline_v2.py'],
           'code_files_hash': pr.code_files_hash(['models/baseline_v2.py'], HERE),
           'notes': ['評価期間(2026-09-01〜10-01)の結果は作るときに使っていない。ただし soft-v2 の表は、評価期間に含まれる手動判定40レースの照合を使っている(基準モデルと本モデルに同じだけ効く)',
                     '成功確率・1号艇・足の評価は出さない(null)']}
    with open(os.path.join(HERE, 'versions', 'baseline-v2.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(man, f, ensure_ascii=False, indent=1); f.write('\n')
    pr.check_manifest(man); print('書きました:', list(files) + ['versions/baseline-v2.json'])


if __name__ == '__main__':
    main()
