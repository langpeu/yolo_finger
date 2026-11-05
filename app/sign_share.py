# 수어 학습기 by Langpeu - (c) 2026 Langpeu, licensed under AGPL-3.0 (see LICENSE)
# Sharing recorded clips between people:
#   export -> signdata_<ID>_<date>.zip : every clip of the 50 shared words + "none" (yours AND merged ones),
#             each clip keeps the temporary ID of the person who recorded it
#   merge  -> zip files from others are added to your word / clip list, then 학습 uses everything.
#             clips are identified by a hash of their content, so a clip is never added twice
#             (same zip merged again, or your own clips coming back inside someone else's zip)
# A clip is joint-coordinate data only (no video, no images).
import io, os, re, json, time, zipfile, hashlib, secrets
import numpy as np

import sign_common as sc

FORMAT = 'yolo_finger-signdata'
FORMAT_VERSION = 1
FEAT_LAYOUT = 'pose 7 points xy (nose, shoulders, elbows, wrists) + 2 hands x 21 points xy + 2 hand flags'
LOW_HAND = 0.5                 # word clips with hands visible less than this are not exported
MAX_ZIP_BYTES = 100 * 1024 * 1024
MAX_UNZIPPED = 500 * 1024 * 1024
MAX_CLIP_BYTES = 5 * 1024 * 1024
MAX_ENTRIES = 20000
ZIP_CLIP_RE = re.compile(r'^clips/([a-z0-9_]{1,32})/([a-z0-9-]{2,24})__([0-9a-f]{16})\.npz$')
CONSENT_TEXT = ('내보내는 데이터는 관절 좌표(영상·사진 없음), 단어, 녹화 날짜, 임시 ID예요. '
                '이 파일을 받은 사람이 합쳐서 학습하고 공유하는 데 동의해요.')


# ---------------------------------------------------------------- shared word list
def load_vocab():
    with open(sc.VOCAB_PATH, encoding='utf-8') as f:
        words = json.load(f)['words']
    return [{'id': 'none', 'name': '(동작 없음)', 'category': '', 'record_sec': 3.0}] + words


def _vocab_maps():
    vocab = load_vocab()
    by_id = {w['id']: w for w in vocab}
    alias = {}
    for w in vocab:
        alias[w['id']] = w['id']
        for a in w.get('aliases', []):
            alias[a] = w['id']
    return by_id, alias


# ---------------------------------------------------------------- my temporary ID
def my_id():
    d = sc._read_json(sc.CONTRIB_PATH, {})
    cid = d.get('id') if isinstance(d, dict) else None
    if not (isinstance(cid, str) and sc.CONTRIB_RE.match(cid)):
        cid = 'u-' + secrets.token_hex(4)
        sc._write_json(sc.CONTRIB_PATH, {'id': cid})
    return cid


def set_my_id(cid):
    cid = (cid or '').strip().lower()
    if not sc.CONTRIB_RE.match(cid):
        raise ValueError('임시 ID는 영어 소문자, 숫자, - 로 2~24자예요. 실명이나 이메일은 쓰지 마세요.')
    sc._write_json(sc.CONTRIB_PATH, {'id': cid})
    return cid


def _hash(seq):
    return hashlib.sha1(np.ascontiguousarray(seq, np.float32).tobytes()).hexdigest()[:16]


_HASH_CACHE = {}      # (word, file) -> content hash; clips never change after saving


def clip_hash(word_id, fname):
    k = (word_id, fname)
    if k not in _HASH_CACHE:
        if sc.clip_contributor(fname):
            _HASH_CACHE[k] = fname.split('__', 1)[1].split('.')[0]   # merged clips are named by their hash
        else:
            _HASH_CACHE[k] = _hash(sc.load_clip(word_id, fname)[0])
    return _HASH_CACHE[k]


def known_hashes():
    out = set()
    for w in sc.load_words():
        for f in sc.clip_files(w['id']):
            try:
                out.add(clip_hash(w['id'], f))
            except Exception:  # noqa: BLE001
                pass
    return out


def _recorded_ms(fname):
    try:
        return int(fname.split('.')[0])
    except ValueError:
        return 0


# ---------------------------------------------------------------- overview for the web page
def overview():
    by_id, alias = _vocab_maps()
    me = my_id()
    words, excluded, pool = [], [], {}
    for w in sc.load_words():
        files = sc.clip_files(w['id'])
        own = [f for f in files if sc.clip_contributor(f) is None]
        for f in files:
            c = sc.clip_contributor(f)
            if c:
                pool.setdefault(c, {'clips': 0, 'words': set()})
                pool[c]['clips'] += 1
                pool[c]['words'].add(w['id'])
        vid = alias.get(w['id'])
        if vid is None:
            if own:
                excluded.append({'id': w['id'], 'name': w['name'], 'clips': len(own)})
            continue
        words.append({'id': w['id'], 'vocab_id': vid, 'name': w['name'], 'own': len(own),
                      'merged': len(files) - len(own)})
    return {
        'me': me,
        'vocab': [{'id': v['id'], 'name': v['name'], 'category': v.get('category', '')} for v in load_vocab()],
        'export_words': words,
        'excluded': excluded,
        'pool': [{'id': c, 'clips': v['clips'], 'words': len(v['words'])} for c, v in sorted(pool.items())],
        'consent_text': CONSENT_TEXT,
    }


# ---------------------------------------------------------------- export
def export_zip():
    """zip bytes + file name: every clip of the shared words (yours and merged), each with its recorder's ID.
    word clips with too few hands are left out; a clip appears once even if it exists twice."""
    by_id, alias = _vocab_maps()
    me = my_id()
    s = sc.load_settings()
    buf = io.BytesIO()
    entries, words, low, seen, people = [], {}, 0, set(), {}
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for w in sc.load_words():
            vid = alias.get(w['id'])
            if vid is None:
                continue
            for f in sc.clip_files(w['id']):
                if not f.endswith('.npz'):
                    continue
                owner = sc.clip_contributor(f) or me
                try:
                    seq, sec = sc.load_clip(w['id'], f)
                except Exception:  # noqa: BLE001
                    continue
                if seq.ndim != 2 or seq.shape[1] != sc.FEAT_DIM:
                    continue
                hand = sc.hand_ratio(seq)
                if vid != 'none' and hand < LOW_HAND:
                    low += 1
                    continue
                h = _hash(seq)
                if h in seen:
                    continue
                seen.add(h)
                rec = _recorded_ms(f) if owner == me else 0
                if owner != me:
                    try:
                        with np.load(os.path.join(sc.DATA_DIR, w['id'], f)) as d:
                            rec = int(d['recorded']) if 'recorded' in d else 0
                    except Exception:  # noqa: BLE001
                        rec = 0
                one = io.BytesIO()
                np.savez(one, seq=seq.astype(np.float32), sec=np.float32(sec), recorded=np.int64(rec))
                name = f'clips/{vid}/{owner}__{h}.npz'
                z.writestr(name, one.getvalue())
                people[owner] = people.get(owner, 0) + 1
                entries.append({'file': name, 'word': vid, 'contributor': owner, 'sec': round(float(sec), 2), 'frames': int(len(seq)),
                                'hand': round(hand, 3), 'recorded': time.strftime('%Y-%m-%d', time.localtime(rec / 1000)) if rec else ''})
                words[vid] = words.get(vid, 0) + 1
        if not entries:
            raise ValueError('내보낼 클립이 없어요. 50개 단어 목록에 있는 단어를 녹화해 주세요.')
        manifest = {
            'format': FORMAT, 'format_version': FORMAT_VERSION,
            'feat_dim': sc.FEAT_DIM, 'feat_layout': FEAT_LAYOUT, 'fps': s['fps'],
            'contributor': me, 'contributors': [{'id': k, 'clips': v} for k, v in sorted(people.items())],
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'consent': True, 'consent_text': CONSENT_TEXT,
            'words': [{'id': k, 'name': by_id[k]['name'], 'clips': v} for k, v in sorted(words.items())],
            'clips': entries,
        }
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=1))
    fname = f'signdata_{me}_{time.strftime("%Y%m%d-%H%M")}.zip'
    return buf.getvalue(), fname, {'clips': len(entries), 'words': len(words), 'people': len(people),
                                   'low_skipped': low}


# ---------------------------------------------------------------- merge
def _local_word_for(vid, by_id):
    """the word in my list that stands for shared word `vid` (same ID or an alias); added if missing"""
    alias_of = {a: w['id'] for w in by_id.values() for a in w.get('aliases', [])}
    for w in sc.load_words():
        if w['id'] == vid or alias_of.get(w['id']) == vid:
            return w['id'], False
    v = by_id[vid]
    sc.add_word(vid, v['name'], v.get('record_sec', 3.0))
    return vid, True


def merge_zip(data, src_name=''):
    """adds the clips of one exported zip to my clip list. Returns counts; raises ValueError for a bad file."""
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError('파일이 너무 커요 (100MB 초과).')
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError('zip 파일이 아니에요.') from None
    with z:
        infos = z.infolist()
        if len(infos) > MAX_ENTRIES or sum(i.file_size for i in infos) > MAX_UNZIPPED:
            raise ValueError('파일 안의 데이터가 너무 많아요.')
        try:
            m = json.loads(z.read('manifest.json').decode('utf-8'))
        except KeyError:
            raise ValueError('수어 학습기에서 내보낸 파일이 아니에요 (manifest.json 없음).') from None
        except (ValueError, UnicodeDecodeError):
            raise ValueError('manifest.json 을 읽을 수 없어요.') from None
        if m.get('format') != FORMAT:
            raise ValueError('수어 학습기에서 내보낸 파일이 아니에요.')
        if m.get('format_version') != FORMAT_VERSION or m.get('feat_dim') != sc.FEAT_DIM:
            raise ValueError(f"데이터 형식 버전이 달라요 (파일 {m.get('format_version')}, 이 앱 {FORMAT_VERSION}). 앱을 최신으로 받아 주세요.")
        cid = str(m.get('contributor', ''))
        if not sc.CONTRIB_RE.match(cid):
            raise ValueError('임시 ID가 잘못됐어요.')
        if not m.get('consent'):
            raise ValueError('데이터 제공 동의가 없는 파일이에요.')
        by_id, _ = _vocab_maps()
        me = my_id()
        have = known_hashes()
        added = dup = bad = 0
        new_words, local, people = [], {}, set()
        for info in infos:
            mt = ZIP_CLIP_RE.match(info.filename)
            if not mt:
                continue
            vid, owner, h = mt.group(1), mt.group(2), mt.group(3)
            if vid not in by_id or info.file_size > MAX_CLIP_BYTES:
                bad += 1
                continue
            try:
                with np.load(io.BytesIO(z.read(info)), allow_pickle=False) as d:
                    seq = np.asarray(d['seq'], np.float32)
                    sec = float(d['sec'])
                    rec = int(d['recorded']) if 'recorded' in d else 0
            except Exception:  # noqa: BLE001
                bad += 1
                continue
            if (seq.ndim != 2 or seq.shape[1] != sc.FEAT_DIM or not 4 <= len(seq) <= 600
                    or not np.isfinite(seq).all() or not 0.3 <= sec <= 15):
                bad += 1
                continue
            h = _hash(seq)                  # recomputed: the name in the zip is not trusted
            if h in have:
                dup += 1                    # already have it (also: my own clips coming back)
                continue
            if owner == me:
                dup += 1                    # my clip that I deleted on purpose: don't bring it back
                continue
            if vid not in local:
                local[vid], is_new = _local_word_for(vid, by_id)
                if is_new:
                    new_words.append(by_id[vid]['name'])
            wid = local[vid]
            dest_dir = os.path.join(sc.DATA_DIR, wid)
            os.makedirs(dest_dir, exist_ok=True)
            fname = f'{owner}__{h}.npz'
            np.savez(os.path.join(dest_dir, fname), seq=seq, sec=np.float32(sec), recorded=np.int64(rec))
            _HASH_CACHE[(wid, fname)] = h
            have.add(h)
            people.add(owner)
            added += 1
    return {'file': src_name, 'contributor': cid, 'people': sorted(people), 'added': added, 'duplicate': dup,
            'rejected': bad, 'new_words': new_words}


def remove_contributor(cid):
    """deletes every merged clip of one contributor right away (e.g. when they ask to be taken out)"""
    if not sc.CONTRIB_RE.match(cid or ''):
        raise ValueError('임시 ID가 잘못됐어요.')
    n = 0
    for w in sc.load_words():
        n += sc.delete_clips(w['id'], [f for f in sc.clip_files(w['id']) if sc.clip_contributor(f) == cid])
    return n
