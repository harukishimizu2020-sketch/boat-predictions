"""予測記録(S1)の作成・保存・検証。

設計: README.md(このフォルダ)/ DESIGN_prediction_verification.md 第2章・5.3・5.4 / GATE_stage2to3_20261007.md。
使い方:
  python predrec.py make   --root . --race 24_20261010_05 --deadline 2026-10-10T12:41:00+09:00 \
                           --version trial-v0 --input snapshot.json [--mode live|backtest] [--model-output out.json(backtestのみ)]
  python predrec.py verify --root .   # ハッシュ・台帳の連鎖・締切前か・git の履歴(追記だけか)・モデルの再実行で一致するか
  python predrec.py commit --root .   # 未コミットの記録を git にコミットする(push はしない)

決まり:
  - 1レース・1版につき1記録。上書き・削除はしない(作り直しは新しい版で行う)。
  - 本番(live)の記録は締切前に作る。作成時刻は PC の現在時刻で、指定できない。締切を過ぎた make は拒否する。
  - 本番では版の entry のモデルを必ず実行する(手で作った出力は使えない)。モデルのコードは git にコミット済みであること。
  - 版の設定(versions/<版>.json)とモデルのコード(code_files)は版を作った時点で固定。中身が変わっていたら make を拒否する。
  - モデルの predict(snapshot) は、同じ入力なら必ず同じ出力を返すこと(verify が再実行して確かめる)。
  - ハッシュの正規形は Python の json.dumps(sort_keys・区切り空白なし・ensure_ascii=False・UTF-8)。他の言語で確かめるときは同じ書き方にそろえる。
"""
import argparse, datetime as dt, hashlib, json, os, re, subprocess, sys, tempfile

FORMAT = 'predrec-1'
ORIGIN_CLASSES = ['2まくり', '3まくり', '4まくり', '5まくり', '6まくり', '3まくり差し', 'その他']  # GATE 第2章の起点7分類(この順で固定)
MOVES = ['差し', 'まくり', 'まくり差し']
B1_ATTACK = ['なし', '差し', 'まくり', 'まくり差し']
RACE_ID = re.compile(r'^24_(\d{8})_(\d{2})$')  # 大村(jcd=24)だけ。戸田(02)などは拒否する
JST = dt.timedelta(hours=9)
NDIGITS = 6        # 確率は小数6桁に丸めてから記録する(同率の判定を安定させるため)
PROB_TOL = 1e-5
ALPHA = 0.05       # 想定外の有意水準(設計書5.4)
E2_Z = 1.96        # STの95%予測区間(設計書5.4)
EQ_TOL = 1e-9
HERE = os.path.dirname(os.path.abspath(__file__))


class RecordError(Exception):
    pass


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ---------- ハッシュ ----------
def canon(obj):
    """ハッシュ用の正規形: キー順に並べ、空白なし、UTF-8、NaN禁止"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def roundtrip(obj):
    """JSON に書いて読み戻したときと同じ形にそろえる(整数キーなどで、保存前と保存後のハッシュがずれないように)"""
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, allow_nan=False))
    except ValueError as e:
        raise RecordError('JSON にできない値があります(NaN など): %s' % e)


def sha(b):
    return 'sha256:' + hashlib.sha256(b).hexdigest()


def input_hash(snapshot):
    return sha(canon(snapshot))


def record_hash(rec):
    r = json.loads(json.dumps(rec))
    r.get('integrity', {}).pop('record_hash', None)
    return sha(canon(r))


def manifest_hash(manifest):
    return sha(canon(manifest))


def _file_bytes(path):
    """改行を LF にそろえた中身(Windows の git checkout で CRLF になっても同じハッシュになるように)"""
    with open(path, 'rb') as f:
        return f.read().replace(b'\r\n', b'\n')


def code_files_hash(paths, root):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.replace('\\', '/').encode('utf-8') + b'\0' + hashlib.sha256(_file_bytes(os.path.join(root, p))).digest())
    return 'sha256:' + h.hexdigest()


def tool_hash():
    return sha(_file_bytes(os.path.abspath(__file__)))


def tail_u(rhash, official):
    """想定外の判定で同率の結果を按分する乱数。記録のハッシュとレース後の公式結果の両方から決まるので、
    作る側が記録の中身を変えて狙った値にすることはできない。official: 公式結果(results_all.jsonl の1行)"""
    key = {k: official.get(k) for k in ('jcd', 'hd', 'rno', 'order', 'kimarite', 'entry')}
    return int(hashlib.sha256(rhash.encode() + b':' + canon(key)).hexdigest()[:16], 16) / 16 ** 16


# ---------- 時刻 ----------
def parse_time(s):
    if s.endswith('Z'): s = s[:-1] + '+00:00'  # Python 3.10 の fromisoformat は Z を読めない
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None: raise RecordError('時刻にタイムゾーンがありません: ' + s)
    return t


def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ---------- 自信度・主シナリオ ----------
def mark_for(p, bands):
    """設計書第2章: 自信度は主シナリオ確率の帯から機械的に決める。帯は版ごとに固定(GATE 第5章: 構築期間の3分位)"""
    if p >= bands['hi']: return '◎'
    if p < bands['lo']: return '△'
    return '○'


def main_scenario(origin_p):
    i = max(range(len(ORIGIN_CLASSES)), key=lambda k: (origin_p[k], -k))  # 同率なら並び順で先のもの
    return ORIGIN_CLASSES[i], origin_p[i]


# ---------- 評価用の共通計算(S3 の評価器はこれを使う) ----------
def brier(p, y):
    return sum((a - b) ** 2 for a, b in zip(p, y))


def tail(p, actual_idx, u):
    """設計書5.4: 実際の結果と同じか、より起きにくい結果の確率の合計(同率は u で按分)。< ALPHA なら想定外"""
    pa = p[actual_idx]
    less = sum(x for x in p if x < pa - EQ_TOL)
    eq = sum(x for x in p if abs(x - pa) <= EQ_TOL)
    return less + u * eq


def st_z(mean, sd, actual):
    return (actual - mean) / sd


def st_misread(z):
    """設計書5.4: |z| > 1.96(95%予測区間の外)ならスタートの読み違い(E2)"""
    return abs(z) > E2_Z


# ---------- 検査 ----------
def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _probs(name, d, keys, allow_null=False):
    if not isinstance(d, dict) or set(d) != set(keys): raise RecordError('%s の項目が違います' % name)
    v = [d[k] for k in keys if not (allow_null and d[k] is None)]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0 or x > 1 for x in v): raise RecordError(name + ' に0〜1以外の値')
    if not allow_null and abs(sum(v) - 1) > PROB_TOL: raise RecordError('%s の合計が1ではありません(%.6f)' % (name, sum(v)))


def check_schema(obj, schema_name):
    try:
        import jsonschema
    except ImportError:
        raise RecordError('jsonschema が入っていません(pip install jsonschema)')
    try:
        jsonschema.validate(obj, _load(os.path.join(HERE, 'schema', schema_name + '.schema.json')))
    except jsonschema.exceptions.ValidationError as e:
        raise RecordError('形式の誤り(%s): %s(場所: %s)' % (schema_name, e.message, '/'.join(map(str, e.absolute_path))))


def check_manifest(manifest):
    check_schema(manifest, 'model_version')
    if manifest['origin_classes'] != ORIGIN_CLASSES: raise RecordError('版の起点の分類・並び順が predrec と違います')
    if manifest['bands']['lo'] > manifest['bands']['hi']: raise RecordError('版の帯が lo > hi です')
    if manifest['entry'] not in manifest['code_files']: raise RecordError('版の entry が code_files に入っていません')


def check_record(rec, manifest):
    """スキーマで書けない決まりを確かめる"""
    check_schema(rec, 'prediction')
    m = RACE_ID.match(rec['race']['race_id'])
    if not m: raise RecordError('大村(24_YYYYMMDD_RR)以外のレースIDです')
    if rec['race']['hd'] != m.group(1) or rec['race']['rno'] != int(m.group(2)): raise RecordError('race_id と hd/rno が食い違っています')
    if rec['model']['version'] != manifest['version']: raise RecordError('版が違います')
    if rec['model']['manifest_hash'] != manifest_hash(manifest): raise RecordError('版の設定ファイルが記録時と違います')
    if rec['model']['code_files_hash'] != manifest['code_files_hash']: raise RecordError('記録のコードのハッシュが版の設定と違います')
    rp, ip = paths_for(rec['race']['race_id'], manifest['version'])
    if rec['input']['path'] != ip: raise RecordError('入力ファイルの置き場所が決まりと違います')
    if rec['origin']['classes'] != ORIGIN_CLASSES: raise RecordError('起点の分類・並び順が違います')
    p = rec['origin']['p']
    if abs(sum(p) - 1) > PROB_TOL: raise RecordError('起点の確率の合計が1ではありません')
    o, pm = main_scenario(p)
    ms = rec['main_scenario']
    if ms['origin'] != o or ms['p'] != pm: raise RecordError('主シナリオが起点の確率の最大と一致しません')
    c = rec['confidence']
    if c['bands'] != manifest['bands'] or c['mark'] != mark_for(pm, manifest['bands']): raise RecordError('自信度が版の帯から決まる値と違います')
    pr = rec['premise']
    cs = pr['entry']['courses']
    if not all(_is_int(x) for x in cs) or sorted(cs) != [1, 2, 3, 4, 5, 6]: raise RecordError('進入の予想が1〜6の並びになっていません')
    for name in ('st', 'form'):
        if pr.get(name) is not None:
            bs = [s['boat'] for s in pr[name]]
            if not all(_is_int(x) for x in bs) or sorted(bs) != [1, 2, 3, 4, 5, 6]: raise RecordError('%s の予想が6艇そろっていません' % name)
    if rec.get('boats') is not None:
        bs = [b['boat'] for b in rec['boats']]
        if not all(_is_int(x) for x in bs) or sorted(bs) != [2, 3, 4, 5, 6]: raise RecordError('動きの予測は2〜6号艇の5艇')
        for b in rec['boats']:
            _probs('%d号艇の動き' % b['boat'], b['move_p'], MOVES)
            if b['success_p'] is not None: _probs('%d号艇の成功' % b['boat'], b['success_p'], MOVES, allow_null=True)
            if b['boat'] == 2 and b['move_p']['まくり差し'] > 0: raise RecordError('2号艇のまくり差しは起きない(内側が1号艇だけ)')
    if rec.get('boat1') is not None: _probs('1号艇の攻められ方', rec['boat1']['attacked_p'], B1_ATTACK)
    if rec['mode'] == 'live':
        d, cr, cap = rec['race']['deadline'], rec['created_at'], rec['input']['captured_at']
        if d is None: raise RecordError('本番の記録には締切時刻が必要です')
        if parse_time(d).utcoffset() != JST: raise RecordError('締切時刻は日本時間(+09:00)で書きます')
        if cap is None: raise RecordError('本番の記録には入力の取得時刻(captured_at)が必要です')
        if not (parse_time(cap) <= parse_time(cr) < parse_time(d)): raise RecordError('時刻の順が「取得 ≤ 作成 < 締切」になっていません')
        if parse_time(cr) < parse_time(manifest['frozen_at']): raise RecordError('版を固定する前に作られた記録です')
        if rec['model']['output_source'] != 'entry': raise RecordError('本番の記録は版のモデルを実行して作ります')
        if rec['model']['code_commit'] is None: raise RecordError('本番の記録にはモデルのコードの git コミットが必要です(未コミットの変更が無いこと)')
    if rec['integrity']['record_hash'] != record_hash(rec): raise RecordError('記録のハッシュが合いません(書き換えの疑い)')


# ---------- 作成・保存 ----------
def load_manifest(root, version):
    p = os.path.join(root, 'versions', version + '.json')
    if not os.path.exists(p): raise RecordError('版の設定がありません: ' + p)
    m = _load(p)
    check_manifest(m)
    return m


def paths_for(race_id, version):
    hd = RACE_ID.match(race_id).group(1)
    return ('records/24/%s/%s__%s.json' % (hd, race_id, version),
            'inputs/24/%s/%s__%s.input.json' % (hd, race_id, version))


def run_model(manifest, snapshot, root):
    """版の設定の entry(predict(snapshot) を持つファイル)を読み込んで実行する"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('predrec_model_' + re.sub(r'\W', '_', manifest['version']), os.path.join(root, manifest['entry']))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.predict(roundtrip(snapshot))


def _r(x):
    return None if x is None else round(float(x), NDIGITS)


def normalize_output(mo):
    """モデルの出力を記録の形にそろえる(確率は小数6桁、艇番は整数)"""
    mo = roundtrip(mo)
    try:
        origin = [_r(mo['origin'][c]) for c in ORIGIN_CLASSES]
        pr = mo['premise']
        out = {
            'premise': {
                'entry': {'courses': [int(x) for x in pr['entry']['courses']], 'p_as_predicted': _r(pr['entry'].get('p_as_predicted'))},
                'st': None if pr.get('st') is None else [{'boat': int(s['boat']), 'mean': float(s['mean']), 'sd': float(s['sd'])} for s in pr['st']],
                'form': None if pr.get('form') is None else [dict(f, boat=int(f['boat'])) for f in pr['form']],
            },
            'origin': origin,
            'scenario_id': mo.get('scenario_id'),
            'boats': None if mo.get('boats') is None else [
                {'boat': int(b['boat']), 'move_p': {k: _r(b['move_p'][k]) for k in MOVES},
                 'success_p': None if b.get('success_p') is None else {k: _r(b['success_p'].get(k)) for k in MOVES}}
                for b in mo['boats']],
            'boat1': None if mo.get('boat1') is None else {
                'attacked_p': {k: _r(mo['boat1']['attacked_p'][k]) for k in B1_ATTACK}, 'escape_p': _r(mo['boat1'].get('escape_p'))},
            'rationale': mo.get('rationale', []),
        }
    except (KeyError, TypeError, ValueError) as e:
        raise RecordError('モデルの出力の形が違います: %r' % e)
    return out


def _git(root, *args):
    try:
        return subprocess.run(['git', '-C', root] + list(args), capture_output=True, text=True, encoding='utf-8', check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def code_commit(manifest, root):
    """モデルのコードがコミット済みならその HEAD を返す。git でない・未コミットの変更があるときは None"""
    head = _git(root, 'rev-parse', 'HEAD')
    if head is None: return None
    if _git(root, 'status', '--porcelain', '--', *manifest['code_files'], 'versions/%s.json' % manifest['version']): return None
    if _git(root, 'ls-files', '--error-unmatch', *manifest['code_files']) is None: return None
    return head.strip()


def make_record(snapshot, manifest, race_id, deadline, mode='live', deadline_source=None, code_root=HERE, model_output=None):
    """記録を組み立てる。model_output を省くと版のモデルを実行する(本番は必ずこちら)。
    派生項目(主シナリオ・自信度・ハッシュ・作成時刻)はここで計算し、呼び出し側からは渡せない"""
    m = RACE_ID.match(race_id)
    if not m: raise RecordError('大村(24_YYYYMMDD_RR)以外のレースIDです: ' + race_id)
    if mode not in ('live', 'backtest'): raise RecordError('mode は live か backtest')
    if mode == 'live' and model_output is not None: raise RecordError('本番の記録は版のモデルを実行して作ります(--model-output は使えない)')
    snapshot = roundtrip(snapshot)
    ch = code_files_hash(manifest['code_files'], code_root)
    if ch != manifest['code_files_hash']: raise RecordError('モデルのコードが版の固定時と変わっています。新しい版を作ってください')
    source = 'file' if model_output is not None else 'entry'
    mo = normalize_output(model_output if model_output is not None else run_model(manifest, snapshot, code_root))
    o, pm = main_scenario(mo['origin'])
    rec = {
        'format': FORMAT,
        'mode': mode,
        'race': {'race_id': race_id, 'jcd': '24', 'hd': m.group(1), 'rno': int(m.group(2)),
                 'deadline': deadline, 'deadline_source': deadline_source},
        'created_at': now_utc(),
        'model': {'version': manifest['version'], 'manifest_hash': manifest_hash(manifest), 'code_files_hash': ch,
                  'code_commit': code_commit(manifest, code_root), 'output_source': source, 'tool_hash': tool_hash()},
        'input': {'hash': input_hash(snapshot), 'path': paths_for(race_id, manifest['version'])[1],
                  'captured_at': snapshot.get('captured_at')},
        'premise': mo['premise'],
        'origin': {'classes': ORIGIN_CLASSES, 'p': mo['origin']},
        'main_scenario': {'origin': o, 'p': pm, 'scenario_id': mo['scenario_id']},
        'confidence': {'mark': mark_for(pm, manifest['bands']), 'bands': manifest['bands']},
        'boats': mo['boats'],
        'boat1': mo['boat1'],
        'rationale': mo['rationale'],
        'integrity': {'record_hash': None},
    }
    if mode == 'live' and deadline and dt.datetime.now(dt.timezone.utc) >= parse_time(deadline): raise RecordError('締切を過ぎています')
    rec['integrity']['record_hash'] = record_hash(rec)
    check_record(rec, manifest)
    return rec, snapshot


def _ledger_path(root):
    return os.path.join(root, 'ledger.jsonl')


def _read_ledger(root):
    p = _ledger_path(root)
    if not os.path.exists(p): return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_new(path, obj):
    """新規ファイルとして書く(一時ファイルに書いてから置き換える。既にあれば拒否)"""
    if os.path.exists(path): raise RecordError('既に記録があります(上書きしない): ' + path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, allow_nan=False); f.write('\n')
    os.replace(tmp, path)


def save(rec, snapshot, root):
    """記録と入力を新規ファイルとして書き、台帳に1行足す。台帳の各行は前の行のハッシュを持つ(抜き取り・入れ替えの検出)。
    同時に2つ動かないよう、台帳のロックファイルを使う"""
    lock = os.path.join(root, 'ledger.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RecordError('別の make が動いています(残っていれば ledger.lock を確認して消す)')
    try:
        os.close(fd)
        rp, ip = paths_for(rec['race']['race_id'], rec['model']['version'])
        for p in (rp, ip):
            if os.path.exists(os.path.join(root, p)): raise RecordError('既に記録があります(上書きしない): ' + p)
        led = _read_ledger(root)
        prev = sha(canon(led[-1])) if led else None
        _write_new(os.path.join(root, ip), snapshot)
        _write_new(os.path.join(root, rp), rec)
        line = {'seq': len(led) + 1, 'race_id': rec['race']['race_id'], 'version': rec['model']['version'], 'mode': rec['mode'],
                'created_at': rec['created_at'], 'record_hash': rec['integrity']['record_hash'], 'path': rp, 'prev': prev}
        with open(_ledger_path(root), 'a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + '\n')
        return rp
    finally:
        os.remove(lock)


# ---------- 検証 ----------
def _replay_problem(rec, snap, manifest, root):
    """版のモデルを入力で実行し直し、記録と同じ予測になるか(手で書き換えていないか)"""
    if code_files_hash(manifest['code_files'], root) != manifest['code_files_hash']:
        return 'モデルのコードが版の固定時と変わっていて、実行し直して確かめられない'
    mo = normalize_output(run_model(manifest, snap, root))
    got = {'premise': mo['premise'], 'origin': mo['origin'], 'scenario_id': mo['scenario_id'],
           'boats': mo['boats'], 'boat1': mo['boat1'], 'rationale': mo['rationale']}
    want = {'premise': rec['premise'], 'origin': rec['origin']['p'], 'scenario_id': rec['main_scenario']['scenario_id'],
            'boats': rec['boats'], 'boat1': rec['boat1'], 'rationale': rec['rationale']}
    return None if canon(got) == canon(want) else 'モデルを実行し直した結果と記録の予測が違う'


def _git_history_problems(root):
    """台帳が追記だけで育っているか、記録・入力が一度も消されていないか(git の履歴で確かめる)"""
    probs = []
    log = _git(root, 'log', '--reverse', '--format=%H', '--', 'ledger.jsonl')
    if log:
        prev = ''
        for h in log.split():
            cur = _git(root, 'show', '%s:ledger.jsonl' % h)
            cur = '' if cur is None else cur.replace('\r\n', '\n')
            if not cur.startswith(prev): probs.append('台帳: コミット %s で過去の行が書き換え・削除されている' % h[:10])
            prev = cur
        if os.path.exists(_ledger_path(root)):
            with open(_ledger_path(root), encoding='utf-8') as f: now = f.read().replace('\r\n', '\n')
            if not now.startswith(prev): probs.append('台帳: コミット済みの行が書き換え・削除されている(未コミットの変更)')
    dl = _git(root, 'log', '--diff-filter=DR', '--name-only', '--format=', '--', 'records', 'inputs')
    for p in sorted(set((dl or '').split())): probs.append(p + ': 一度コミットした記録・入力が削除・移動されている')
    return probs


def verify(root, replay=True):
    """問題の一覧を返す(空なら合格)"""
    probs = []
    try:
        led = _read_ledger(root)
    except ValueError as e:
        return ['台帳が JSON として読めない: %s' % e]
    prev, seen, manifests = None, set(), {}
    in_git = _git(root, 'rev-parse', '--is-inside-work-tree') is not None
    for i, line in enumerate(led):
        tag = '台帳%d行目(%s)' % (i + 1, line.get('race_id'))
        if line.get('seq') != i + 1: probs.append(tag + ': 連番が飛んでいる')
        if line.get('prev') != prev: probs.append(tag + ': 前の行のハッシュが合わない(抜き取り・入れ替えの疑い)')
        prev = sha(canon(line))
        try:
            rec = _load(os.path.join(root, line['path']))
            race_id, v = rec['race']['race_id'], rec['model']['version']
            if (line['race_id'], line['version'], line['mode']) != (race_id, v, rec['mode']): probs.append(tag + ': 台帳と記録の中身(レース・版・mode)が違う')
            if line['path'] != paths_for(race_id, v)[0]: probs.append(tag + ': 記録ファイルの置き場所が決まりと違う')
            if (race_id, v) in seen: probs.append(tag + ': 同じレース・同じ版の記録が2つある')
            seen.add((race_id, v))
            if v not in manifests: manifests[v] = load_manifest(root, v)
            check_record(rec, manifests[v])
            if rec['integrity']['record_hash'] != line['record_hash']: probs.append(tag + ': 台帳と記録のハッシュが違う')
            snap = _load(os.path.join(root, rec['input']['path']))
            if input_hash(snap) != rec['input']['hash']: probs.append(tag + ': 入力ファイルが記録時と違う')
            elif replay and rec['model']['output_source'] == 'entry':
                pr = _replay_problem(rec, snap, manifests[v], root)
                if pr: probs.append(tag + ': ' + pr)
        except (OSError, ValueError, KeyError, TypeError, RecordError) as e:
            probs.append('%s: %r' % (tag, e)); continue
        if in_git:
            log = _git(root, 'log', '--format=%H %cI', '--', line['path'])
            if log and log.strip():
                commits = log.strip().splitlines()
                if len(commits) > 1: probs.append(tag + ': 記録ファイルがコミット後に変更されている')
                if rec['mode'] == 'live' and parse_time(commits[-1].split()[1]) >= parse_time(rec['race']['deadline']):
                    probs.append(tag + ': 最初のコミットが締切後')
            elif rec['mode'] == 'live' and dt.datetime.now(dt.timezone.utc) >= parse_time(rec['race']['deadline']):
                probs.append(tag + ': 締切を過ぎてもコミットされていない')
    paths = {l.get('path') for l in led}
    for dp, _, fs in os.walk(os.path.join(root, 'records')):
        for f in fs:
            p = os.path.relpath(os.path.join(dp, f), root).replace('\\', '/')
            if p not in paths: probs.append(p + ': 台帳に無いファイル')
    if in_git: probs += _git_history_problems(root)
    return probs


def commit(root):
    if _git(root, 'rev-parse', '--is-inside-work-tree') is None: raise RecordError('git のリポジトリではありません: ' + root)
    _git(root, 'add', '--', 'records', 'inputs', 'ledger.jsonl')
    st = _git(root, 'diff', '--cached', '--name-only')
    if not st or not st.strip(): return None
    n = sum(1 for x in st.split() if x.startswith('records/'))
    if _git(root, 'commit', '-m', '予測記録 %d件(%s)' % (n, now_utc())) is None: raise RecordError('git commit に失敗しました(git の user.name / user.email が未設定の可能性)')
    return _git(root, 'rev-parse', 'HEAD').strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('make'); a.add_argument('--root', default='.'); a.add_argument('--race', required=True)
    a.add_argument('--deadline'); a.add_argument('--deadline-source'); a.add_argument('--version', required=True)
    a.add_argument('--input', required=True); a.add_argument('--mode', default='live', choices=['live', 'backtest'])
    a.add_argument('--model-output', help='backtest だけ。手で用意したモデルの出力を使う')
    b = sub.add_parser('verify'); b.add_argument('--root', default='.'); b.add_argument('--no-replay', action='store_true')
    c = sub.add_parser('commit'); c.add_argument('--root', default='.')
    x = ap.parse_args()
    try:
        if x.cmd == 'make':
            man = load_manifest(x.root, x.version)
            rec, snap = make_record(_load(x.input), man, x.race, x.deadline, x.mode, deadline_source=x.deadline_source,
                                    code_root=x.root, model_output=_load(x.model_output) if x.model_output else None)
            print(save(rec, snap, x.root), rec['main_scenario']['origin'], rec['main_scenario']['p'], rec['confidence']['mark'])
        elif x.cmd == 'verify':
            pr = verify(x.root, replay=not x.no_replay)
            print('\n'.join(pr) if pr else '問題なし(%d件)' % len(_read_ledger(x.root)))
            sys.exit(1 if pr else 0)
        else:
            print(commit(x.root) or 'コミットするものはありません')
    except RecordError as e:
        print('拒否: %s' % e, file=sys.stderr); sys.exit(2)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print('エラー: %r' % e, file=sys.stderr); sys.exit(3)


if __name__ == '__main__':
    main()
