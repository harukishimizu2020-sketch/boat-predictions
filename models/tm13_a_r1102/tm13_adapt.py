"""snapshot → 本モデルの入力 inp への読み替え。
2つの形を読む。
  1) 本番 daily_live の snapshot(venues24_20261031/SNAPSHOT_FORMAT.md、before.boats がある形)
  2) 学習に使った aux_compact と同じ形(before.racers・originalTenjis がある形。試験用)
本番の形は、学習の値(aux)と同じ意味になるように直す:
  - 展示タイムの順位 tenjiRank = 6艇のタイムの昇順の順位(同じなら同順位。aux 40,002/40,002 レースで一致)
  - スタート展示の順位 startTenjiRank = フライングの艇を先に(Fの大きさの小さい順)、次にそれ以外を ST の小さい順(同じなら同順位)。
    出遅れ(late)の艇は ST 0 として数に入れ、自分の順位は無し(aux 120,933 レースで確かめた規則。例外 0)
  - スタート展示の ST はフライングを負の値(aux と同じ)
  - 進入 startSinnyu = start_exhibition.by_course の course(会場ページの venue_exhibition_course は使わない)
  - オリジナル展示は isshu・mawariashi・chokusen だけ(桐生の hanshu は半周で別の量。学習でも桐生の一周は欠け)
  - 気象は boatrace.jp の表示(前のレース時点。学習の aux も同じ表示)
  - race_class(is-ippan など)は本番の snapshot に無い → snap['race_class'] があれば使い、無ければ欠け(NaN)
"""


def _bymap(lst, key):
    out = {}
    for x in lst or []:
        if x and x.get(key) is not None: out[int(x[key])] = x
    return out


def race_info(snap):
    k = str(snap.get('key') or snap.get('race_id'))
    tob = snap.get('tobans')
    if not tob:
        tob = {str(int(c['boat'])): str(c.get('toban') or c.get('racerNumber') or '') for c in (snap.get('card') or [])
               if c.get('boat') is not None and not c.get('absent')}
    return {'key': k, 'jcd': k[:2], 'hd': k[3:11], 'rno': int(k[12:]), 'tobans': {b: t for b, t in tob.items() if t}}


def is_live(snap):
    return isinstance((snap.get('before') or {}).get('boats'), dict)


def _num(v):
    if v is None or v == '': return None
    try: return float(v)
    except (TypeError, ValueError): return None


def live_racers(bef):
    rac = {}
    for k, x in (bef.get('boats') or {}).items():
        if not x: continue
        b = int(k)
        rac[b] = {'boatNumber': b, 'tenjiTime': _num(x.get('tenji_time')), 'tilt': _num(x.get('tilt')),
                  'weight': _num(x.get('weight')), 'weightAdjust': _num(x.get('weight_adjust')),
                  'partsExchange': x.get('parts_exchange') or None}
    tt = {b: r['tenjiTime'] for b, r in rac.items() if r.get('tenjiTime') not in (None, 0)}
    for b, v in tt.items(): rac[b]['tenjiRank'] = 1 + sum(1 for y in tt.values() if y < v)
    keys = {}; late = set()
    for e in ((bef.get('start_exhibition') or {}).get('by_course')) or []:
        try: b = int(e.get('boat'))
        except (TypeError, ValueError): continue
        r = rac.setdefault(b, {'boatNumber': b})
        if e.get('course') is not None: r['startSinnyu'] = int(e['course'])
        st = _num(e.get('st'))
        if e.get('late'):
            r['startTenjiTime'] = 0.0; keys[b] = (1, 0.0); late.add(b)
        elif st is not None:
            fl = bool(e.get('flying')) or st < 0
            r['startTenjiTime'] = -abs(st) if fl else st
            keys[b] = (0 if fl else 1, abs(st))
    for b, k in keys.items():
        if b not in late: rac[b]['startTenjiRank'] = 1 + sum(1 for y in keys.values() if y < k)
    return rac


def live_ot(bef):
    ot = {}
    for k, x in (bef.get('boats') or {}).items():
        o = (x or {}).get('orig') or {}
        d = {'isshuTime': _num(o.get('isshu')), 'mawariashiTime': _num(o.get('mawariashi')), 'chokusenTime': _num(o.get('chokusen'))}
        if any(v is not None for v in d.values()): ot[int(k)] = d
    return ot


def live_weather(bef):
    w = bef.get('weather') or {}
    wd = w.get('wind_direction_code')
    try: wd = int(wd) if wd not in (None, '') else None
    except (TypeError, ValueError): wd = None
    out = {'windSpeed': _num(w.get('wind_speed_m')), 'windDirection': wd, 'waveHeight': _num(w.get('wave_cm')),
           'weatherDegree': _num(w.get('air_temp_c')), 'waterDegree': _num(w.get('water_temp_c')), 'weather': w.get('weather') or None}
    return {k: v for k, v in out.items() if v is not None}


CARD_KEYS = ('grade', 'F', 'L', 'startAvg', 'winAll', 'win2All', 'win3All', 'winLocal', 'win2Local', 'win3Local', 'motor2', 'motor3')


def live_card(snap):
    card = {}
    for c in snap.get('card') or []:
        if c.get('boat') is None or c.get('absent'): continue
        card[int(c['boat'])] = {k: c.get(k) for k in CARD_KEYS}
    return card


def to_inp(snap):
    info = race_info(snap); bef = snap.get('before') or {}
    base = {'race_id': info['key'], 'jcd': info['jcd'], 'hd': info['hd'], 'rno': info['rno'],
            'history': snap.get('tm13_history') or {}, 'program': snap.get('program')}
    if is_live(snap):
        base.update({'card': live_card(snap), 'before': live_weather(bef), 'racers': live_racers(bef), 'ot': live_ot(bef),
                     'race_class': snap.get('race_class') or snap.get('raceClass')})
    else:
        base.update({'card': _bymap(snap.get('card'), 'boat'), 'before': bef,
                     'racers': _bymap(bef.get('racers'), 'boatNumber'),
                     'ot': _bymap(snap.get('originalTenjis') if snap.get('originalTenjis') is not None else bef.get('originalTenjis'), 'boatNumber'),
                     'race_class': snap.get('raceClass')})
    return base
