"""24場の締切前予測を毎日まわす本体(2026-10-31 下書き。毎日の自動運用はまだ始めていない)。
設計: /mnt/project-files/venues24_20261031/DAILY_DESIGN.md(要件 F1〜F8)。記録の作り・検証は predrec.py のまま使う。

使い方(PC で、公開リポジトリ prediction_record の中から):
  python tools/daily_live.py plan   --state <状態フォルダ> --fetcher <取得モジュール.py> [--hd YYYYMMDD]
  python tools/daily_live.py tick   --state ... --fetcher ... --version <版[,版...]> [--lead-min 8] [--push] [--ctx '{...}']
  python tools/daily_live.py loop   --state ... --fetcher ... --version <版> [--push]   (plan のあと、全レースが済むまで1分ごとに tick、最後に close)
  python tools/daily_live.py close  --state ... --version <版> [--hd YYYYMMDD]   (記録にも漏れにも無いまま締切を過ぎたレースを漏れに入れる)
  python tools/daily_live.py sample --state ... --version <版> --results <公式結果.jsonl> --hd YYYYMMDD
状態フォルダ(plan_*.json・misses_*.jsonl・runlog.jsonl・sample_*.json)は公開リポジトリの外に置く。

取得モジュール(会場の公式サイトの読み方。kyotei_mark1/venues24_20261031/fetch/ を包む)は次の2つを持つ:
  racelist(hd) -> [{'race_id': 'JJ_YYYYMMDD_RR', 'deadline': 'YYYY-MM-DDTHH:MM:00+09:00', 'deadline_source': url, 'card': {...} または省く,
                     'race_class': 'is-ippan' など(boatrace.jp の節の格、任意), 'race_title', 'meet_day'(任意)}]
  before(race_id) -> {'captured_at': UTC の ISO 時刻, 'sources': [url...], 'raw': {...}, 'normalized': {...}}
 任意で card(race_id) -> {...}(出走表)。racelist が card を省いたレースは、tick で記録を作る直前に card() で取る
 (2026-11-01 PC の試運転: 朝に全レースの出走表を取ると boatrace.jp が1回約10秒で plan に約32分かかり、終わる前に16レースの締切が過ぎた。
  racelist は会場ごとに締切時刻の一覧だけを取り、出走表は tick で取る形を勧める)。
決まり:
  - 締切の lead 分前から締切の margin 秒前(既定90秒)までに、直前情報を取り、版のモデルで記録を作る。取れない・間に合わないレースは
    misses に理由つきで残し、後から予測を足さない(締切後は作らない。predrec も拒否する)。
  - 採点の標本は記録の中身を見ずに、レースIDだけで決める(sha256(種|レースID) が rate 未満、always の会場は全部)。
    種と率は版の設定 sample_rule に書き、版を固定したら変えない。
"""
import argparse, datetime as dt, hashlib, importlib.util, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import predrec as pr
import evaluate as E

UTC = dt.timezone.utc


def now():
    return dt.datetime.now(UTC)


def jst_today():
    return (now() + pr.JST).strftime('%Y%m%d')


def load_fetcher(path):
    spec = importlib.util.spec_from_file_location('daily_fetcher', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for f in ('racelist', 'before'):
        if not hasattr(m, f): raise SystemExit('取得モジュールに %s が無い: %s' % (f, path))
    return m


def _jl_append(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8', newline='\n') as f: f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + '\n')


def _jl(path):
    if not os.path.exists(path): return []
    with open(path, encoding='utf-8') as f: return [json.loads(l) for l in f if l.strip()]


def plan_path(state, hd): return os.path.join(state, 'plan_%s.json' % hd)
def miss_path(state, hd): return os.path.join(state, 'misses_%s.jsonl' % hd)
def log_path(state): return os.path.join(state, 'runlog.jsonl')


def plan(state, fetcher, hd):
    """F1: その日の全会場・全レースの締切時刻と出走表。既にあれば作り直さない(締切時刻を後から動かさない)"""
    p = plan_path(state, hd)
    if os.path.exists(p): return pr._load(p)
    races = sorted(fetcher.racelist(hd), key=lambda r: (r['deadline'], r['race_id']))
    for r in races:
        jcd, rhd, _ = pr.parse_race_id(r['race_id'])
        if rhd != hd: raise pr.RecordError('別の日のレースが混ざっている: ' + r['race_id'])
        if r.get('deadline') is None or pr.parse_time(r['deadline']).utcoffset() != pr.JST:
            raise pr.RecordError('締切時刻が無いか日本時間でない: ' + r['race_id'])
    obj = {'hd': hd, 'made_at': now().replace(microsecond=0).isoformat(), 'races': races}
    os.makedirs(state, exist_ok=True)
    pr._write_new(p, obj)
    _jl_append(log_path(state), {'at': obj['made_at'], 'event': 'plan', 'hd': hd, 'n': len(races),
                                 'venues': sorted({r['race_id'][:2] for r in races})})
    return obj


def done_ids(root, state, hd, version):
    rec = {l['race_id'] for l in pr._read_ledger(root) if l['version'] == version and l['race_id'][3:11] == hd}
    miss = {m['race_id'] for m in _jl(miss_path(state, hd)) if m['version'] == version}
    return rec, miss


def preflight(root):
    """記録を作る前に、締切前の commit が確実にできるかを確かめる(作ってから commit で失敗すると、締切後に未コミットの記録が残る)"""
    if pr._git(root, 'rev-parse', '--is-inside-work-tree') is None: raise pr.RecordError('git のリポジトリではありません: ' + root)
    for k in ('user.name', 'user.email'):
        if not (pr._git(root, 'config', k) or '').strip():
            raise pr.RecordError('git の %s が未設定です。リポジトリの中で git config %s "..." を設定してから動かす' % (k, k))


RACE_META = ('race_class', 'race_title', 'meet_day')   # racelist(hd) の各レースに任意で付ける属性 → snapshot の同じ名前の欄


_RESULTS = {}


def load_results_store(path, hd, years=2):
    """snapshot_builder に渡す過去の公式結果(races_core と同じ形の行)。その日より前・直近 years 年だけ。ファイルの更新時刻で再読込"""
    if not path: return None
    import gzip
    st = os.stat(path).st_mtime
    key = (path, hd, years)
    if key in _RESULTS and _RESULTS[key][0] == st: return _RESULTS[key][1]
    lo = '%04d%s' % (int(hd[:4]) - years, hd[4:])
    op = gzip.open if path.endswith('.gz') else open
    rows = []
    with op(path, 'rt', encoding='utf-8') as f:
        for l in f:
            if not l.strip(): continue
            o = json.loads(l)
            if lo <= str(o.get('hd', '')) < hd: rows.append(o)   # 当日以降は渡さない(未来の結果を入力に混ぜない)
    rows.sort(key=lambda o: (o['hd'], o.get('jcd', ''), int(o.get('rno') or 0)))
    _RESULTS.clear(); _RESULTS[key] = (st, rows)
    return rows


def load_builder(root, manifest):
    """版の設定 snapshot_builder(code_files に入れたファイル)の build(snap, ctx) を読む。無ければ None。
    記録の前に snapshot に履歴などを足す関数(足した後の snapshot をハッシュして残すので、replay も同じ入力で確かめられる)"""
    path = manifest.get('snapshot_builder')
    if not path: return None
    if path not in manifest['code_files']: raise pr.RecordError('snapshot_builder は版の code_files に入れる: ' + path)
    spec = importlib.util.spec_from_file_location('snap_builder_' + manifest['version'].replace('-', '_').replace('.', '_'), os.path.join(root, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.build


def _versions(version):
    return [v for v in (version.split(',') if isinstance(version, str) else list(version)) if v]


def tick(root, state, fetcher, version, hd=None, lead_min=8.0, margin_s=90, push=False, t=None, clock=None, commit_buffer_s=15, ctx=None):
    """F2〜F5: 締切が近いレースを締切の順に記録する。1回の呼び出しで、いま取るべきレースを全部処理する。
    version はカンマ区切りで複数の版にできる(取得は1回、記録は版ごと。基準の版と本モデルの版を同じ入力で並べるため)。
    時刻はレースごとに今の時刻で確かめ、記録は1レースごとにすぐ commit(と push)する
    (2026-11-01 PC の試運転: 最後にまとめて commit したら、締切39秒前に作った記録の commit が締切の14秒後になった)。
    取得が締切の commit_buffer_s 秒前までかかったら、記録を作らずに漏れにする。t は試験用の固定時刻、clock は試験用の時計。
    ctx は snapshot_builder に渡す辞書。ctx に results_store(ファイルの場所)があれば、その日より前の結果の行を ctx['results'] に入れて渡す"""
    clock = clock or ((lambda: t) if t else now)
    hd = hd or jst_today(); t0 = clock()
    preflight(root)
    vs = _versions(version)
    mans = {v: pr.load_manifest(root, v) for v in vs}
    builders = {v: load_builder(root, mans[v]) for v in vs}
    pl = plan(state, fetcher, hd)
    done = {v: done_ids(root, state, hd, v) for v in vs}
    made, miss_new, waiting, commits, pushed = [], [], 0, [], None

    def miss(rid, v, reason, at):
        m = {'race_id': rid, 'version': v, 'reason': reason, 'at': at.isoformat()}
        _jl_append(miss_path(state, hd), m); miss_new.append(m)

    for r in pl['races']:
        rid = r['race_id']
        todo = [v for v in vs if rid not in done[v][0] and rid not in done[v][1] and rid[:2] in pr.venues_of(mans[v])]
        if not todo: continue
        d = pr.parse_time(r['deadline']); cur = clock()
        if cur >= d - dt.timedelta(seconds=margin_s):
            for v in todo: miss(rid, v, '締切に間に合わなかった(取得の時間帯に処理できなかった)', cur)
            continue
        if cur < d - dt.timedelta(minutes=lead_min):
            waiting += 1; continue
        try:
            card = r.get('card')
            if card is None and hasattr(fetcher, 'card'): card = fetcher.card(rid)
            b = fetcher.before(rid)
        except Exception as e:  # 取れなかったら次の tick でもう一度(締切の margin 秒前までに取れなければ misses)
            # 取り直しのレースは「待ち」に数える(数えないと loop が終わり、記録にも漏れにも残らないレースが出る。2026-11-01 下関12R)
            waiting += 1
            _jl_append(log_path(state), {'at': clock().isoformat(), 'event': 'retry', 'race_id': rid, 'error': repr(e)[:300]})
            continue
        base = {'key': rid, 'captured_at': b['captured_at'], 'card': card, 'before': b.get('normalized'),
                'raw': b.get('raw'), 'sources': b.get('sources')}
        for k in RACE_META:   # racelist で取れたレースの属性(節の格など)。取れていない欄は入れない
            if r.get(k) is not None: base[k] = r[k]
        any_made = False
        for v in todo:
            try:
                snap = json.loads(json.dumps(base))
                if builders[v]:
                    c2 = dict(ctx or {}, hd=hd, race_id=rid, state=state)
                    if c2.get('results_store'): c2['results'] = load_results_store(c2['results_store'], hd)
                    snap = builders[v](snap, c2)
                cur = clock()
                if cur >= d - dt.timedelta(seconds=commit_buffer_s):
                    miss(rid, v, '取得が締切の直前までかかった(%d秒前を切った)' % commit_buffer_s, cur); continue
                rec, sn = pr.make_record(snap, mans[v], rid, r['deadline'], 'live', deadline_source=r.get('deadline_source'), code_root=root)
                pr.save(rec, sn, root); made.append(rid if len(vs) == 1 else '%s@%s' % (rid, v)); any_made = True
            except Exception as e:
                waiting += 1
                _jl_append(log_path(state), {'at': clock().isoformat(), 'event': 'retry', 'race_id': rid, 'version': v, 'error': repr(e)[:300]})
        if not any_made: continue
        c = pr.commit(root)   # 1レースごとにすぐ。失敗は RecordError で止める(preflight を通っていれば起きない想定)
        if c: commits.append(c)
        if c and push:
            ok = pr._git(root, 'push', '-q', 'origin', 'HEAD') is not None
            pushed = ok if pushed is None else (pushed and ok)
            if not ok: _jl_append(log_path(state), {'at': clock().isoformat(), 'event': 'push_failed', 'race_id': rid})
    _jl_append(log_path(state), {'at': t0.isoformat(), 'end': clock().isoformat(), 'event': 'tick', 'hd': hd, 'versions': vs, 'made': len(made),
                                 'missed': len(miss_new), 'waiting': waiting, 'commits': len(commits), 'pushed': pushed})
    return {'made': made, 'missed': miss_new, 'waiting': waiting, 'commit': commits[-1] if commits else None,
            'commits': len(commits), 'pushed': pushed}


def close_day(root, state, version, hd, t=None):
    """その日の終わりに、記録にも漏れにも無いまま締切を過ぎたレースを漏れに入れる(どのレースも必ずどちらかに残す)。version はカンマ区切り可"""
    t = t or now()
    pl = pr._load(plan_path(state, hd)) if os.path.exists(plan_path(state, hd)) else {'races': []}
    out = []
    for v in _versions(version):
        man = pr.load_manifest(root, v)
        recd, missed = done_ids(root, state, hd, v)
        venues = set(pr.venues_of(man))
        for r in pl['races']:
            rid = r['race_id']
            if rid in recd or rid in missed or rid[:2] not in venues: continue
            if t >= pr.parse_time(r['deadline']):
                m = {'race_id': rid, 'version': v, 'reason': '処理されないまま締切を過ぎた(取得の取り直し中・処理の停止など)', 'at': t.isoformat()}
                _jl_append(miss_path(state, hd), m); out.append(m)
    _jl_append(log_path(state), {'at': t.isoformat(), 'event': 'close_day', 'hd': hd, 'missed': len(out)})
    return out


def sample(state, version_manifest, results, hd):
    """F7: 採点の標本。記録の中身を見ずに、レースIDと版の sample_rule だけで決める"""
    rule = version_manifest.get('sample_rule') or {}
    seed, rate, always = rule.get('seed'), float(rule.get('rate', 0)), set(rule.get('always', []))
    if not seed: raise pr.RecordError('版に sample_rule.seed が無い(標本の規則は版を固定する前に決める)')
    venues = set(pr.venues_of(version_manifest))
    picks = []
    for k, o in sorted(results.items()):
        if o.get('hd') != hd or str(o.get('jcd')).zfill(2) not in venues or not E.race_valid(o)[0]: continue
        u = int(hashlib.sha256(('%s|%s' % (seed, k)).encode()).hexdigest()[:16], 16) / 2 ** 64
        if k[:2] in always or u < rate: picks.append({'race_id': k, 'u': round(u, 6), 'why': 'always' if k[:2] in always else 'rate'})
    out = {'hd': hd, 'version': version_manifest['version'], 'rule': rule, 'n': len(picks), 'picks': picks}
    p = os.path.join(state, 'sample_%s__%s.json' % (hd, version_manifest['version']))
    if not os.path.exists(p): pr._write_new(p, out)
    return pr._load(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['plan', 'tick', 'loop', 'close', 'sample'])
    ap.add_argument('--root', default='.'); ap.add_argument('--state', required=True); ap.add_argument('--fetcher')
    ap.add_argument('--version'); ap.add_argument('--hd'); ap.add_argument('--results')
    ap.add_argument('--lead-min', type=float, default=8.0); ap.add_argument('--margin-s', type=int, default=90)
    ap.add_argument('--push', action='store_true')
    ap.add_argument('--ctx', help='snapshot_builder に渡す JSON(例 {"results_dir": "..."})')
    a = ap.parse_args()
    ctx = json.loads(a.ctx) if a.ctx else None
    hd = a.hd or jst_today()
    if a.cmd == 'sample':
        print(json.dumps(sample(a.state, pr.load_manifest(a.root, a.version), E.load_results(a.results), hd)['n']))
        return
    if a.cmd == 'close':
        print(len(close_day(a.root, a.state, a.version, hd))); return
    f = load_fetcher(a.fetcher)
    if a.cmd == 'plan':
        print(len(plan(a.state, f, hd)['races'])); return
    while True:
        r = tick(a.root, a.state, f, a.version, hd, a.lead_min, a.margin_s, a.push, ctx=ctx)
        print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in r.items()}, ensure_ascii=False), flush=True)
        if a.cmd == 'tick': break
        if r['waiting'] == 0 and not r['made']:
            close_day(a.root, a.state, a.version, hd); break   # loop の終わりに、残ったレースを漏れにする
        time.sleep(60)


if __name__ == '__main__':
    main()
