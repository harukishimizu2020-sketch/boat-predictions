"""24場の締切前予測を毎日まわす本体(2026-10-31 下書き。毎日の自動運用はまだ始めていない)。
設計: /mnt/project-files/venues24_20261031/DAILY_DESIGN.md(要件 F1〜F8)。記録の作り・検証は predrec.py のまま使う。

使い方(PC で、公開リポジトリ prediction_record の中から):
  python tools/daily_live.py plan   --state <状態フォルダ> --fetcher <取得モジュール.py> [--hd YYYYMMDD]
  python tools/daily_live.py tick   --state ... --fetcher ... --version <版> [--lead-min 8] [--push]
  python tools/daily_live.py loop   --state ... --fetcher ... --version <版> [--push]   (plan のあと、全レースが済むまで1分ごとに tick)
  python tools/daily_live.py sample --state ... --version <版> --results <公式結果.jsonl> --hd YYYYMMDD
状態フォルダ(plan_*.json・misses_*.jsonl・runlog.jsonl・sample_*.json)は公開リポジトリの外に置く。

取得モジュール(会場の公式サイトの読み方。kyotei_mark1/venues24_20261031/fetch/ を包む)は次の2つを持つ:
  racelist(hd) -> [{'race_id': 'JJ_YYYYMMDD_RR', 'deadline': 'YYYY-MM-DDTHH:MM:00+09:00', 'deadline_source': url, 'card': {...}}]
  before(race_id) -> {'captured_at': UTC の ISO 時刻, 'sources': [url...], 'raw': {...}, 'normalized': {...}}
決まり:
  - 締切の lead 分前から締切の margin 秒前までに、直前情報を取り、版のモデルで記録を作る。取れない・間に合わないレースは
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


def tick(root, state, fetcher, version, hd=None, lead_min=8.0, margin_s=60, push=False, t=None):
    """F2〜F5: 締切が近いレースを記録する。1回の呼び出しで、いま取るべきレースを全部処理する"""
    hd = hd or jst_today(); t = t or now()
    man = pr.load_manifest(root, version)
    pl = plan(state, fetcher, hd)
    recd, missed = done_ids(root, state, hd, version)
    venues = set(pr.venues_of(man))
    made, miss_new, waiting = [], [], 0
    for r in pl['races']:
        rid = r['race_id']
        if rid in recd or rid in missed or rid[:2] not in venues: continue
        d = pr.parse_time(r['deadline'])
        if t >= d - dt.timedelta(seconds=margin_s):
            m = {'race_id': rid, 'version': version, 'reason': '締切に間に合わなかった(取得の時間帯に処理できなかった)', 'at': t.isoformat()}
            _jl_append(miss_path(state, hd), m); miss_new.append(m); continue
        if t < d - dt.timedelta(minutes=lead_min):
            waiting += 1; continue
        try:
            b = fetcher.before(rid)
            snap = {'key': rid, 'captured_at': b['captured_at'], 'card': r.get('card'), 'before': b.get('normalized'),
                    'raw': b.get('raw'), 'sources': b.get('sources')}
            rec, s = pr.make_record(snap, man, rid, r['deadline'], 'live', deadline_source=r.get('deadline_source'), code_root=root)
            pr.save(rec, s, root); made.append(rid)
        except Exception as e:  # 取れなかったら次の tick でもう一度(締切の margin 秒前までに取れなければ misses)
            _jl_append(log_path(state), {'at': t.isoformat(), 'event': 'retry', 'race_id': rid, 'error': repr(e)[:300]})
    commit = pr.commit(root) if made else None
    pushed = None
    if commit and push:
        pushed = pr._git(root, 'push', '-q', 'origin', 'HEAD') is not None
    _jl_append(log_path(state), {'at': t.isoformat(), 'event': 'tick', 'hd': hd, 'made': len(made), 'missed': len(miss_new),
                                 'waiting': waiting, 'commit': commit, 'pushed': pushed})
    return {'made': made, 'missed': miss_new, 'waiting': waiting, 'commit': commit, 'pushed': pushed}


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
    ap.add_argument('cmd', choices=['plan', 'tick', 'loop', 'sample'])
    ap.add_argument('--root', default='.'); ap.add_argument('--state', required=True); ap.add_argument('--fetcher')
    ap.add_argument('--version'); ap.add_argument('--hd'); ap.add_argument('--results')
    ap.add_argument('--lead-min', type=float, default=8.0); ap.add_argument('--margin-s', type=int, default=60)
    ap.add_argument('--push', action='store_true')
    a = ap.parse_args()
    hd = a.hd or jst_today()
    if a.cmd == 'sample':
        print(json.dumps(sample(a.state, pr.load_manifest(a.root, a.version), E.load_results(a.results), hd)['n']))
        return
    f = load_fetcher(a.fetcher)
    if a.cmd == 'plan':
        print(len(plan(a.state, f, hd)['races'])); return
    while True:
        r = tick(a.root, a.state, f, a.version, hd, a.lead_min, a.margin_s, a.push)
        print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in r.items()}, ensure_ascii=False), flush=True)
        if a.cmd == 'tick' or r['waiting'] == 0 and not r['made']: break
        time.sleep(60)


if __name__ == '__main__':
    main()
