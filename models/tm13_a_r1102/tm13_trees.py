"""LightGBM の木を純 Python で評価する(lightgbm 不要)。木は train_a.py が dump_model から平らな配列にして保存したもの。
判定の規則は LightGBM の NumericalDecision と同じ: 欠損の扱い(missing_type None/Zero/NaN)と default_left。"""
import math
K_ZERO = 1e-35


def flatten(tree_structure):
    """dump_model の木 → (feat, thr, dl, mt, left, right, leafval)。葉は負の番号 ~i"""
    feat, thr, dl, mt, left, right, leaves = [], [], [], [], [], [], []
    MT = {'None': 0, 'Zero': 1, 'NaN': 2}
    def rec(n):
        if 'leaf_value' in n:
            leaves.append(n['leaf_value']); return ~(len(leaves) - 1)
        i = len(feat); feat.append(n['split_feature']); thr.append(n['threshold']); dl.append(1 if n['default_left'] else 0)
        mt.append(MT[n['missing_type']]); left.append(None); right.append(None)
        assert n.get('decision_type', '<=') == '<='
        left[i] = rec(n['left_child']); right[i] = rec(n['right_child']); return i
    root = rec(tree_structure)
    return [feat, thr, dl, mt, left, right, leaves, root]


def tree_value(t, x):
    feat, thr, dl, mt, left, right, leaves, root = t
    if root < 0: return leaves[~root]
    i = root
    while i >= 0:
        v = x[feat[i]]; m = mt[i]
        nan = v != v
        if nan and m != 2: v = 0.0
        if (m == 1 and -K_ZERO <= v <= K_ZERO) or (m == 2 and nan):
            i = left[i] if dl[i] else right[i]
        else:
            i = left[i] if v <= thr[i] else right[i]
    return leaves[~i]


def predict_proba(model, x):
    """model = {'num_class', 'trees'}。softmax の確率"""
    K = model['num_class']; s = [0.0] * K
    for j, t in enumerate(model['trees']): s[j % K] += tree_value(t, x)
    mx = max(s); e = [math.exp(v - mx) for v in s]; z = sum(e)
    return [v / z for v in e]
