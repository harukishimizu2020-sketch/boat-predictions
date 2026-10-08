"""S4 本モデル(s4-v1)の過去レースでの試し(バックテスト)の記録をまとめて作る。tools/backtest.py の s4 版。
本番の記録とは混ぜない: 公開リポジトリとは別のフォルダ(--root)に作る。記録は mode=backtest になり、段階2→3の判定には数えない。
入力(スナップショット)は tools/build_s4.py の Data.snapshot(レース前の情報 + その日より前の判定済みレース・公式結果からの履歴)。
オッズと本番STは入れない。出走表が無いレース(除外レースなど)も記録は作る(card などが None → 中央値で埋まる。評価では除外される)。
使い方: python tools/backtest_s4.py --root <別フォルダ> --version s4-v1 --results <results_all.jsonl> --extract <_gate_extract.json> \
                                   --results-long <results_omura_long.jsonl> --from 20260901 --to 20261001
  --root には versions/・models/・schema/・labels/ を写し、git にコミットしておく。"""
import argparse, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tools'))
import predrec as pr
import build_s4 as S


def main():
    ap = argparse.ArgumentParser()
    for k in ('--root', '--version', '--results', '--extract', '--from', '--to'): ap.add_argument(k, required=True)
    ap.add_argument('--results-long')
    ap.add_argument('--labels-omura', default=os.path.join(HERE, 'labels', 'soft-v2.json'))
    ap.add_argument('--labels-other', default=os.path.join(HERE, 'labels', 'soft-v1.json'))
    a = ap.parse_args()
    man = pr.load_manifest(a.root, a.version)
    data = S.Data(a.results, a.extract, a.results_long, a.labels_omura, a.labels_other)
    keys = sorted(k for k, r in data.res.items() if str(r['jcd']) == S.OMURA and getattr(a, 'from') <= r['hd'] <= a.to)
    n = 0
    for k in keys:
        snap = data.snapshot(k)
        assert snap['history_cutoff'] == data.res[k]['hd']
        rec, snap = pr.make_record(snap, man, k, None, 'backtest', code_root=a.root)
        pr.save(rec, snap, a.root); n += 1
    print('記録しました: %d レース(%s〜%s)' % (n, getattr(a, 'from'), a.to))


if __name__ == '__main__':
    main()
