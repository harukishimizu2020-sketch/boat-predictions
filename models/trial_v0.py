"""S1 の動作確認用モデル trial-v0(判定に使わない)。
全レースに同じ起点の確率を出す。値は kyotei_mark1/data/analysis/analysis5_origin.md「起点の割合」のソフト列
(大村485レース・2026-07-02〜10-01の全期間、2026-10-07のソフトラベル修正後)。
S3 の基準モデルは、構築期間だけで割合を計算し直して別の版にする(評価期間を含むこの値は基準に使わない)。"""

ORIGIN = {'2まくり': 0.097, '3まくり': 0.318, '4まくり': 0.088, '5まくり': 0.088,
          '6まくり': 0.024, '3まくり差し': 0.270, 'その他': 0.115}


def predict(snapshot):
    return {
        'origin': dict(ORIGIN),
        'premise': {'entry': {'courses': [1, 2, 3, 4, 5, 6], 'p_as_predicted': None}, 'st': None, 'form': None},
        'scenario_id': None,
        'boats': None,
        'boat1': None,
        'rationale': [],
    }
