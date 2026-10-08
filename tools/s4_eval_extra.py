"""S4 の評価の補助(S4_REPORT.md 用)。evaluate.py が出さない2つを計算する。
 (1) 艇ごとの動きの Brier(ソフト・ハード): バックテストの記録(--root)の move_p と、判定ラベルを soft-v2 で直した正解を比べる。
     版ごと(s4-v1・baseline-v2)に、数えたレース(枠なり・除外なし・判定あり)だけ。
 (2) 参考(探索): 選ばなかった候補(B など)を同じ学習データで学習し、評価期間の Brier を出す。主結果ではない(S4_DESIGN.md 2.3)。
使い方: python tools/s4_eval_extra.py --root <バックテストのフォルダ> --results <results_all.jsonl> --extract <_gate_extract.json> \
          --results-long <results_omura_long.jsonl> --judge <judge_omura.json> --from 20260901 --to 20261001 --out <JSON>"""
import argparse, importlib.util, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tools'))
import predrec as pr
import evaluate as E
import build_s4 as S

OC = pr.ORIGIN_CLASSES; MOVES = pr.MOVES
EXPLORE = [('A', 1.0, 1.0), ('A', 1.0, 0.5), ('A', 0.3, 0.7), ('A', 0.1, 0.4), ('B', 1.0, 0.8), ('B', 1.0, 1.0), ('B', 0.3, 0.5), ('B', 0.1, 0.3)]


def move_brier(root, version, res, jd, lab, keys):
    out = {b: {'n': 0, 'soft': 0.0, 'hard': 0.0} for b in range(2, 7)}
    for seq, rec in E.load_records(root, version):
        k = rec['race']['race_id']
        if k not in keys: continue
        r = res[k]
        for pb in rec['boats']:
            b = pb['boat']; t, c = E.judge_label(jd[k], b)
            v = E.soft_move(b, t, c, lab, r)
            if v is None: continue  # 判断不可は数えない
            p = [pb['move_p'][m] for m in MOVES]; y = [1.0 if m == t else 0.0 for m in MOVES]
            out[b]['n'] += 1; out[b]['soft'] += pr.brier(p, v); out[b]['hard'] += pr.brier(p, y)
    return {str(b): {'n': d['n'], 'brier_soft': d['soft'] / d['n'], 'brier_hard': d['hard'] / d['n']} for b, d in out.items() if d['n']}


def main():
    ap = argparse.ArgumentParser()
    for k in ('--root', '--results', '--extract', '--judge', '--from', '--to', '--out'): ap.add_argument(k, required=True)
    ap.add_argument('--results-long')
    a = ap.parse_args()
    lab = E.load_labels(os.path.join(HERE, 'labels', 'soft-v2.json'))
    res = E.load_results(a.results); jd = E.load_judge(a.judge)
    keys = {k for k, r in res.items() if str(r['jcd']) == '24' and getattr(a, 'from') <= r['hd'] <= a.to and E.race_valid(r)[0] and isinstance(jd.get(k), dict)}
    out = {'n_scored': len(keys), 'move_brier': {v: move_brier(a.root, v, res, jd, lab, keys) for v in ('s4-v1', 'baseline-v2')}}
    for b in range(2, 7):
        s, bl = out['move_brier']['s4-v1'][str(b)], out['move_brier']['baseline-v2'][str(b)]
        s['bss_soft'] = (bl['brier_soft'] - s['brier_soft']) / bl['brier_soft']; s['bss_hard'] = (bl['brier_hard'] - s['brier_hard']) / bl['brier_hard']
    # (2) 参考(探索)
    data = S.Data(a.results, a.extract, a.results_long, os.path.join(HERE, 'labels', 'soft-v2.json'), os.path.join(HERE, 'labels', 'soft-v1.json'))
    mod = S.load_model_module(); params = mod.load_params(); fp = params['feature_params']
    rows = []
    for k in data.training_keys():
        L = data.labels[k]; _, vals = mod.features(data.snapshot(k), fp)
        rows.append({'key': k, 'x': [np.nan if v is None else v for v in vals], 'q': L['q'], 'P': L['P'], 'hard': L['hard']})
    Xraw = np.array([r['x'] for r in rows], float); sc = S.scaler(Xraw); X = S.transform(Xraw, sc)
    ev = sorted(k for k in keys if k in data.labels and k in data.aux)
    Xe = S.transform(np.array([[np.nan if v is None else v for v in mod.features(data.snapshot(k), fp)[1]] for k in ev], float), sc)
    q = np.array([data.labels[k]['q'] for k in ev]); days = [data.labels[k]['hd'] for k in ev]
    base = np.array([[params['base_origin'][c] for c in OC]] * len(ev))
    b_base = S.brier_rows(base, q)
    fits = {}
    expl = {'n_eval': len(ev), 'base_brier_soft': float(b_base.mean()), 'configs': {}}
    for cand, alpha, lam in EXPLORE:
        if alpha not in fits: fits[alpha] = S.fit_all(X, rows, alpha)
        A, B = fits[alpha]
        pm = S.predict_proba(Xe, A) if cand == 'A' else S.origin_from_moves(S.moves_proba(Xe, B))
        p = (1 - lam) * base + lam * pm
        bs = S.brier_rows(p, q)
        ci = E.bss_ci(list(b_base), list(bs)); cid = E.bss_ci(list(b_base), list(bs), days)
        expl['configs']['%s|%g|%g' % (cand, alpha, lam)] = {'brier_soft': float(bs.mean()), 'bss': ci['bss'], 'ci_race': [ci['lo'], ci['hi']], 'ci_day': [cid['lo'], cid['hi']],
                                                             'note': '主結果(s4-v1 と同じ設定。evaluate.py の値と一致するはず)' if (cand, alpha, lam) == ('A', 1.0, 1.0) else '参考(探索)'}
    out['explore'] = expl
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f: json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
