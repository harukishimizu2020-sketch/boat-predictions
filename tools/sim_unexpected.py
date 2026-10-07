"""条件2(想定外)の数え方を模擬計算で比べる(S3_REPORT.md 6章 判断A の注意点の確認)。
判定AIのラベルの誤り方を、正解ラベルの表(labels/soft-v1.json)と構築期間のラベルの出方から作り、
「本当に較正されたモデル」と「過信したモデル」で、300件・600件の判定が過信と出る割合を数え方ごとに求める。

前提(この模擬計算の限界):
  - 表の値(ラベル→本当の動きの確率)が正しい。艇どうしの判定の誤りは独立。
  - 本当の動きは艇ごとに独立に、表と構築期間のラベルの出方から逆算した割合で起きる。
  - 1着艇の決まり手でラベルが確定する規則は使わない(着順を作らないため)。
使い方: python tools/sim_unexpected.py --results results_omura.jsonl --judge judge_omura.json [--reps 2000] [--seed 20261007]"""
import argparse, itertools, os, random, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools'))
import predrec as pr
import evaluate as E
from build_baseline import races, OMURA_FROM, SPLIT

OC = pr.ORIGIN_CLASSES; MOVES = pr.MOVES; CONFS = ('高', '中', '低')


def label_model(res, jd, lab):
    """艇ごとに、ラベル (t, c) の出方 P(t,c)、本当の動きの割合 π(m)、本当の動き m のときにラベル (t, c) が付く確率 L(t,c|m)"""
    con = races(res, jd, OMURA_FROM, SPLIT)
    out = {}
    for b in range(2, 7):
        cnt = {}
        for k, r in con:
            t, c = E.judge_label(jd[k], b)
            if t in MOVES and c in CONFS: cnt[(t, c)] = cnt.get((t, c), 0) + 1
        n = sum(cnt.values())
        P = {tc: v / n for tc, v in cnt.items()}
        Q = {tc: E.soft_move(b, tc[0], tc[1], lab, None) for tc in P}
        pi = [sum(P[tc] * Q[tc][m] for tc in P) for m in range(3)]
        L = {m: [(tc, P[tc] * Q[tc][m] / pi[m]) for tc in sorted(P)] for m in range(3) if pi[m] > 0}
        out[b] = {'pi': pi, 'L': L, 'n': n}
    return out, len(con)


def origin_dist(pis):
    """艇ごとに独立に動くときの起点7分類の確率(本当に較正されたモデル)"""
    p = [0.0] * len(OC)
    for ms in itertools.product(range(3), repeat=5):
        w = 1.0
        for b, m in zip(range(2, 7), ms): w *= pis[b][m]
        if w == 0: continue
        p[OC.index(E.origin_hard({b: MOVES[m] for b, m in zip(range(2, 7), ms)}))] += w
    return p


def draw(rng, items):
    x = rng.random(); s = 0.0
    for v, w in items:
        s += w
        if x < s: return v
    return items[-1][0]


def simulate(lm, lab, p, n, reps, seed):
    rng = random.Random(seed)
    crit = E.crit_unexpected(n)
    pis = {b: lm[b]['pi'] for b in lm}
    mitems = {b: [(m, pis[b][m]) for m in range(3)] for b in lm}
    hit = {'true': 0, 'hard': 0, 'soft': 0}; mean = {'true': 0.0, 'hard': 0.0, 'soft': 0.0}
    for _ in range(reps):
        k = {'true': 0, 'hard': 0, 'soft': 0.0}
        for _ in range(n):
            truth, labels, P = {}, {}, {}
            for b in range(2, 7):
                m = draw(rng, mitems[b]); truth[b] = MOVES[m]
                t, c = draw(rng, lm[b]['L'][m]); labels[b] = t
                P[b] = E.soft_move(b, t, c, lab, None)
            u = rng.random()
            tails = [pr.tail(p, ci, u) < E.ALPHA for ci in range(len(OC))]
            k['true'] += tails[OC.index(E.origin_hard(truth))]
            k['hard'] += tails[OC.index(E.origin_hard(labels))]
            q = E.origin_soft(P)
            k['soft'] += sum(qc for qc, f in zip(q, tails) if f)
        for key in hit:
            hit[key] += k[key] >= crit; mean[key] += k[key] / reps
    return {'n': n, 'crit': crit, 'reps': reps, 'rate_over_crit': {k: v / reps for k, v in hit.items()},
            'mean_count': mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True); ap.add_argument('--judge', required=True)
    ap.add_argument('--labels', default=os.path.join(HERE, 'labels', 'soft-v1.json'))
    ap.add_argument('--reps', type=int, default=2000); ap.add_argument('--seed', type=int, default=20261007)
    ap.add_argument('--powers', type=float, nargs='+', default=[1.0, 1.15, 1.3, 1.5])
    a = ap.parse_args()
    lab = E.load_labels(a.labels); res = E.load_results(a.results); jd = E.load_judge(a.judge)
    lm, ncon = label_model(res, jd, lab)
    p = origin_dist({b: lm[b]['pi'] for b in lm})
    print('構築期間 %d レース。本当の動きの割合(差し, まくり, まくり差し):' % ncon)
    for b in lm: print('  %d号艇' % b, [round(x, 3) for x in lm[b]['pi']])
    print('較正されたモデル', dict(zip(OC, [round(x, 3) for x in p])))
    for g in a.powers:  # 確率を g 乗して正規化 = 過信の度合い(g=1 は較正済み)
        model = [x ** g for x in p]; model = [x / sum(model) for x in model]
        print('g=%s' % g, dict(zip(OC, [round(x, 3) for x in model])))
        for n in (300, 600):
            print('  ', simulate(lm, lab, model, n, a.reps, a.seed + n), flush=True)


if __name__ == '__main__':
    main()
