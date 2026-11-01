"""24場の基準モデル(会場ごとの単純な発生率)を作る下書き(2026-10-31)。判定し直しの完了後に、作り直した統合データでもう一度実行してから版を固定する。
目的: 24場の予測記録で、本モデルと比べる「会場ごとの起点・13類・動き・ST の単純な割合」。毎レース、会場ごとに同じ確率を出す。
決まり(設計メモ https://claude.ai/artifact/LK9t9dp5tBxBwtyQsbHgc5 の基準D・6章、統計的点検 F-4):
  - 作るのに使うのは 2026-08-31 以前だけ(10/19 以降は未使用期間。9/1〜10/18 も使わない)。
  - 会場ごとに作り、判定の少ない会場は全国の割合へ寄せる(寄せる強さ k は、〜6/30 で作って 7/1〜8/31 で測る対数損失が一番小さいものを選ぶ)。
  - タイム(ST)は会場ごとに平均・標準偏差を出し、会場をまたいで混ぜない。
使い方: python tools/build_baseline24.py --core <races_core.jsonl.gz> --repo . --out <出力フォルダ>
  出力: report.json・REPORT.md・baseline24_params.json。models/baseline24_v0_params.json はこの出力を写したもの(2026-11-02 版 baseline24-v0)。"""
import argparse, collections, gzip, json, math, os, sys

BUILD_TO = '20260831'
TUNE_SPLIT = '20260701'   # 〜6/30 で作り、7/1〜8/31 で k を選ぶ
ST_FROM = '20251215'
K_GRID = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 1e9]
FLOOR = 1e-4


def load(core):
    with gzip.open(core, 'rt', encoding='utf-8') as f:
        for l in f:
            if l.strip(): yield json.loads(l)


def cls_of(r):
    c = r.get('cls') or {}
    key = r.get('cls_default') if r.get('cls_default') in c else 'v2'
    return c.get(key)


def vec(d, classes, other='その他'):
    v = [0.0] * len(classes)
    for k, p in (d or {}).items():
        v[classes.index(k) if k in classes else classes.index(other)] += float(p)
    s = sum(v)
    return [x / s for x in v] if s > 0 else None


def mean(vs):
    n = len(vs)
    return [sum(v[i] for v in vs) / n for i in range(len(vs[0]))] if n else None


def shrink(m, n, nat, k):
    if m is None: return list(nat)
    if k >= 1e9: return list(nat)
    p = [(n * a + k * b) / (n + k) for a, b in zip(m, nat)]
    p = [max(x, FLOOR) for x in p]; s = sum(p)
    return [x / s for x in p]


def xent(target, p):
    return -sum(t * math.log(max(q, FLOOR)) for t, q in zip(target, p) if t > 0)


def fit(rows, classes, field, k):
    by = collections.defaultdict(list)
    for r in rows: by[r['jcd']].append(r[field])
    nat = mean([r[field] for r in rows])
    return {j: shrink(mean(v), len(v), nat, k) for j, v in by.items()}, nat


def tune(rows, classes, field):
    tr = [r for r in rows if r['hd'] < TUNE_SPLIT]; te = [r for r in rows if r['hd'] >= TUNE_SPLIT]
    res = {}
    for k in K_GRID:
        m, nat = fit(tr, classes, field, k)
        res[k] = sum(xent(r[field], m.get(r['jcd'], nat)) for r in te) / len(te)
    best = min(res, key=lambda k: (round(res[k], 6), k))
    return best, res, len(tr), len(te)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--core', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import predrec as pr, evaluate as E
    lab = json.load(open(os.path.join(a.repo, 'labels', 'soft-v2.json'), encoding='utf-8'))
    OC, C13 = pr.ORIGIN_CLASSES, pr.CLASS13
    rows, stv = [], collections.defaultdict(lambda: collections.defaultdict(list))
    move = collections.defaultdict(lambda: collections.defaultdict(list))
    n_all = collections.Counter(); n_late = 0
    for r in load(a.core):
        if r['hd'] > BUILD_TO:
            n_late += 1; continue   # 9/1 以降は読まない(数だけ数える)
        if not r.get('valid'): continue
        j = r['jcd']
        if r['hd'] >= ST_FROM:
            for b in range(1, 7):
                s = (r.get('st') or {}).get(str(b))
                try:
                    if s and not str(s).startswith(('F', 'L')): stv[j][b].append(float(s))
                except ValueError: pass
        c = cls_of(r) if r.get('judged') else None
        if not c: continue
        o, t = vec(c.get('origin'), OC), vec(c.get('c13'), C13)
        if o is None or t is None: continue
        n_all[j] += 1
        rows.append({'jcd': j, 'hd': r['hd'], 'o': o, 't': t})
        for b in range(2, 7):
            tt, cc = E.judge_label(r.get('labels'), b)
            v = E.soft_move(b, tt, cc, lab, r)
            if v is not None: move[j][b].append(v)
    out = {'build_to': BUILD_TO, 'tune_split': TUNE_SPLIT, 'n_races_by_venue': dict(sorted(n_all.items())), 'n_rows_after_build_to_skipped': n_late}
    params = {'origin': {}, 'class13': {}, 'move': {}, 'st': {}}
    for field, classes, name in (('o', OC, 'origin'), ('t', C13, 'class13')):
        k, res, ntr, nte = tune(rows, classes, field)
        m, nat = fit(rows, classes, field, k)
        out[name] = {'k': k, 'tune_logloss': {str(x): round(y, 5) for x, y in res.items()}, 'n_tune_train': ntr, 'n_tune_test': nte,
                     'national': dict(zip(classes, [round(x, 6) for x in nat]))}
        params[name] = {j: dict(zip(classes, [round(x, 6) for x in p])) for j, p in sorted(m.items())}
    MV = pr.MOVES
    params['national'] = {'origin': out['origin']['national'], 'class13': out['class13']['national'], 'move': {}, 'st': {}}
    allmv = collections.defaultdict(list); allst = collections.defaultdict(list)
    for j in move:
        for b in range(2, 7): allmv[b] += move[j][b]
    for j in stv:
        for b in range(1, 7): allst[b] += stv[j][b]
    params['national']['move'] = {str(b): dict(zip(MV, [round(x, 6) for x in mean(allmv[b])])) for b in range(2, 7)}
    for b in range(1, 7):
        v = allst[b]; mu = sum(v) / len(v); sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
        params['national']['st'][str(b)] = [round(mu, 4), round(sd, 4), len(v)]
    for j in sorted(move):
        params['move'][j] = {str(b): dict(zip(MV, [round(x, 6) for x in mean(move[j][b])])) for b in range(2, 7) if move[j][b]}
    for j in sorted(stv):
        params['st'][j] = {}
        for b in range(1, 7):
            v = stv[j][b]
            if len(v) >= 30:
                mu = sum(v) / len(v); sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
                params['st'][j][str(b)] = [round(mu, 4), round(sd, 4), len(v)]
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(params, open(os.path.join(a.out, 'baseline24_params.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, sort_keys=True)
    L = ['# 24場の基準モデル(会場ごとの単純な割合)の試算', '',
         '統合データ: `%s`。作るのに使ったのは %s 以前の valid かつ判定済みのレースだけ(以降の %d 行は読んでいない)。' % (a.core, BUILD_TO, n_late),
         '判定し直しの前のラベルでの試算なので、版はまだ固定しない。', '']
    for name in ('origin', 'class13'):
        o = out[name]
        L += ['## %s' % ('起点7分類' if name == 'origin' else '13類'), '',
              '全国へ寄せる強さ k = **%s**(〜6/30 の %d レースで作り、7/1〜8/31 の %d レースで測った対数損失が最小)。' % (o['k'], o['n_tune_train'], o['n_tune_test']), '',
              '| k | 対数損失 |', '|---|---|'] + ['| %s | %.5f |' % ('全国のみ' if float(x) >= 1e9 else x, y) for x, y in o['tune_logloss'].items()] + ['']
    L += ['## 会場ごとの最多の分類', '', '| 会場 | レース数 | 起点の最多 | 13類の最多 |', '|---|---|---|---|']
    for j in sorted(params['origin']):
        po, pt = params['origin'][j], params['class13'][j]
        a1 = max(po, key=po.get); a2 = max(pt, key=pt.get)
        L.append('| %s | %d | %s %.3f | %s %.3f |' % (j, n_all[j], a1, po[a1], a2, pt[a2]))
    open(os.path.join(a.out, 'REPORT.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(json.dumps({'origin_k': out['origin']['k'], 'class13_k': out['class13']['k'], 'venues': len(n_all), 'races': sum(n_all.values())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
