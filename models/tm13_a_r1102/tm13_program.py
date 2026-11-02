"""公式の番組表(B ファイル、前日発行)から候補 I の特徴を作る(純 Python)。accuracy_20261031/scripts/build_bfeat.py と同じ計算。
使い方: progs = parse_program(テキスト, 'YYYYMMDD') → {race_id: {...}}、features(progs[race_id], jcd) → {bi_...: float32}
今節成績は番組表の欄(前の日まで)だけを使う。
ファイル: https://www1.mbrace.or.jp/od2/B/{YYYYMM}/b{yymmdd}.lzh を展開した b{yymmdd}.txt(中身は cp932。racer_lore の fetch_all.py は UTF-8 に直して保存する。
read_program_file はどちらも読む)。1ファイルに全会場の1日分。出走表(racelist)の今節成績は当日の着順も入る恐れがあるので使わない。"""
import json, math, os, re
from tm13_features import f32, NAN

ZEN = str.maketrans('０１２３４５６７８９', '0123456789')
RACE = re.compile(r'^[\s　]*([０-９0-9]+)Ｒ[\s　]+(\S+)')
DAY = re.compile(r'第[\s　]*([０-９0-9]+)日|最終日|初日')
BAD = set('LKS欠失転落妨沈不エ')
HERE = os.path.dirname(os.path.abspath(__file__))
_FAN = None


def fan():
    global _FAN
    if _FAN is None:
        d = json.load(open(os.path.join(HERE, 'tm13_fan.json'), encoding='utf-8'))
        _FAN = {t: 1.0 for t in d['female']}; _FAN.update({t: 0.0 for t in d['male']})
    return _FAN


def _num(s):
    try: return float(s)
    except ValueError: return NAN


def parse_program(text, hd):
    """B ファイル1日分(展開した cp932 を文字列にしたもの)→ {race_id: {'race_name', 'day_label', 'boats': {艇: {...}}}}"""
    out = {}; jcd = None; rno = None; rname = None; day = None
    for l in text.splitlines():
        if re.match(r'^\d{2}BBGN', l): jcd = l[:2]; day = None; continue
        if re.match(r'^\d{2}BEND', l): jcd = None; continue
        if jcd is None: continue
        if day is None:
            d = DAY.search(l)
            if d and 'ボートレース' in l: day = d.group(0); continue
        mr = RACE.match(l)
        if mr: rno = int(mr.group(1).translate(ZEN)); rname = mr.group(2).replace('　', ''); continue
        if rno and re.match(r'^[1-6] \d{4}', l):
            b = l.encode('cp932')
            rest = b[24:65].decode('cp932', 'replace').split()
            ks = b[65:77].decode('cp932', 'replace') if len(b) > 65 else ''
            rid = '%s_%s_%02d' % (jcd, hd, rno)
            e = out.setdefault(rid, {'race_name': rname, 'day_label': day, 'boats': {}})
            e['boats'][int(b[0:1])] = {'toban': b[2:6].decode(), 'age': _num(b[14:16].decode('cp932', 'replace')),
                                       'boat2': _num(rest[7]) if len(rest) >= 8 else NAN, 'ks': ks}
    return out


def _seqsum(v):
    """numpy と同じ順の足し算(Python 3.12 以降の sum() は補正つきで、学習の numpy・pandas と末尾の桁が違う)"""
    s = 0.0
    for x in v: s += x
    return s


def _ks(ks):
    ch = [c for c in ks.replace('　', ' ') if c != ' ']
    fin = [int(c) for c in ch if c in '123456']
    return float(len(ch)), (_seqsum(float(x) for x in fin) / len(fin) if fin else NAN), float(sum(1 for c in fin if c == 1)), \
        float(sum(1 for c in ch if c in BAD)), float(sum(1 for c in ch if c == 'F'))


def _nanmean(v):
    v = [x for x in v if x == x]; return _seqsum(v) / len(v) if v else NAN


def features(entry, jcd=None):
    """I_no_misc の列: 艇ごとの年齢・ボート2連率・今節成績(数・平均着順・1着・事故・F)と年齢・2連率・平均着順の6艇平均との差、
    レースの何日目・優勝戦・準優・予選・特選など・一般、女子の数・6人とも女子"""
    f = {}; B = range(1, 7); bo = (entry or {}).get('boats') or {}
    vals = {}
    for b in B:
        x = bo.get(b) or bo.get(str(b))
        if x:
            x = {k: (NAN if v is None else v) for k, v in x.items()}   # snapshot(JSON)では欠けは None
            n, m, w, bad, F = _ks(x.get('ks') or '')
            vals[b] = {'age': x.get('age', NAN), 'boat2': x.get('boat2', NAN), 'ks_n': n, 'ks_mean': m, 'ks_win': w, 'ks_bad': bad, 'ks_F': F,
                       'female': fan().get(x.get('toban'), NAN)}
        else:
            vals[b] = {k: NAN for k in ('age', 'boat2', 'ks_n', 'ks_mean', 'ks_win', 'ks_bad', 'ks_F', 'female')}
    for c in ('age', 'boat2', 'ks_n', 'ks_mean', 'ks_win', 'ks_bad', 'ks_F'):
        for b in B: f['bi_b%d_%s' % (b, c)] = vals[b][c]
    for c in ('age', 'boat2', 'ks_mean'):
        m = _nanmean([vals[b][c] for b in B])
        for b in B:
            v = vals[b][c]; f['bi_b%d_%s_rl' % (b, c)] = NAN if (v != v or m != m) else v - m
    lbl = (entry or {}).get('day_label')
    if not lbl: f['bi_nday'] = NAN
    elif lbl == '初日': f['bi_nday'] = 1.0
    elif lbl == '最終日': f['bi_nday'] = NAN
    else:
        mm = re.search(r'([０-９0-9]+)', lbl); f['bi_nday'] = float(mm.group(1).translate(ZEN)) if mm else NAN
    rn = (entry or {}).get('race_name') or ''
    f['bi_final'] = float('優勝' in rn and '準優' not in rn); f['bi_semi'] = float('準優' in rn)
    f['bi_yosen'] = float('予選' in rn and '特' not in rn); f['bi_special'] = float(bool(re.search('特選|選抜|ドリーム|特賞', rn)))
    f['bi_ippan'] = float('一般' in rn)
    fem = [vals[b]['female'] for b in B if vals[b]['female'] == vals[b]['female']]
    f['bi_nfem'] = float(sum(fem)) if fem else NAN; f['bi_allfem'] = float(f['bi_nfem'] == 6)
    return {k: f32(v) for k, v in f.items()}


_PCACHE = {}


def read_program_file(path):
    """b{yymmdd}.txt を読んで parse_program の結果を返す(更新時刻が変わるまで使い回す)"""
    st = os.stat(path); key = (path, st.st_mtime, st.st_size)
    if key not in _PCACHE:
        raw = open(path, 'rb').read()
        try: text = raw.decode('utf-8')
        except UnicodeDecodeError: text = raw.decode('cp932', 'replace')
        m = re.search(r'b(\d{6})', os.path.basename(path))
        _PCACHE.clear(); _PCACHE[key] = parse_program(text, '20' + m.group(1))
    return _PCACHE[key]


def for_snapshot(entry, path):
    """snapshot に残す形(JSON にできる値だけ。replay で同じ特徴を作れるように番組表の行の中身をそのまま残す)"""
    if not entry: return None
    def cl(v): return None if isinstance(v, float) and v != v else v
    return {'source': os.path.basename(path), 'race_name': entry.get('race_name'), 'day_label': entry.get('day_label'),
            'boats': {str(b): {k: cl(v) for k, v in x.items()} for b, x in entry['boats'].items()}}
