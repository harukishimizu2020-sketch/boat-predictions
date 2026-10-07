"""過去レースでの試し(バックテスト)の記録をまとめて作る。S3 で評価器を実データに通すためのもの。
本番の記録とは混ぜない: 公開リポジトリとは別のフォルダ(--root)に作る。記録は mode=backtest になり、段階2→3の判定には数えない。
使い方: python tools/backtest.py --root <別フォルダ> --version baseline-v1 --results results_omura.jsonl --from 20260901 --to 20261001
  入力(スナップショット)はレースIDだけ(基準モデルは入力を使わない)。--root には versions/・models/・schema/ を写し、git にコミットしておく。"""
import argparse, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import predrec as pr
import evaluate as E


def main():
    ap = argparse.ArgumentParser()
    for k in ('--root', '--version', '--results', '--from', '--to'): ap.add_argument(k, required=True)
    a = ap.parse_args()
    man = pr.load_manifest(a.root, a.version)
    res = E.load_results(a.results)
    keys = sorted(k for k, r in res.items() if str(r['jcd']) == '24' and getattr(a, 'from') <= r['hd'] <= a.to)
    n = 0
    for k in keys:
        rec, snap = pr.make_record({'race_id': k}, man, k, None, 'backtest', code_root=a.root)
        pr.save(rec, snap, a.root); n += 1
    print('記録しました: %d レース(%s〜%s)' % (n, getattr(a, 'from'), a.to))


if __name__ == '__main__':
    main()
