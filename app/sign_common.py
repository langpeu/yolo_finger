# 수어 학습기 by Langpeu - (c) 2026 Langpeu, licensed under AGPL-3.0 (see LICENSE)
# Shared code: words/settings, clip files, features, closest-person tracker, model, recognizer
import os, re, json, glob, time, math, shutil, collections, urllib.request
os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS', '0')   # faster Media Foundation webcam start
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- folders (all relative to the project folder)
APP_DIR = os.path.dirname(os.path.abspath(__file__))     # <project>/app   : python code
ROOT = os.path.dirname(APP_DIR)                           # <project>
HERE = APP_DIR
DATA_ROOT = os.path.join(ROOT, 'data')                    # <project>/data  : your recordings, model, settings (not in git)
MODELS_DIR = os.path.join(ROOT, 'models')                 # <project>/models: downloaded YOLO / MediaPipe models (not in git)
DATA_DIR = os.path.join(DATA_ROOT, 'clips')               # recorded clips, one folder per word
MODEL_PATH = os.path.join(DATA_ROOT, 'sign_model.pt')
WORDS_PATH = os.path.join(DATA_ROOT, 'words.json')
SETTINGS_PATH = os.path.join(DATA_ROOT, 'settings.json')
HISTORY_PATH = os.path.join(DATA_ROOT, 'train_history.json')
CONTRIB_PATH = os.path.join(DATA_ROOT, 'contributor.json')   # this PC's temporary contributor ID for sharing
DELETED_PATH = os.path.join(DATA_ROOT, 'deleted_clips.json')  # merged clips you deleted / IDs you removed: never merged again
VOCAB_PATH = os.path.join(APP_DIR, 'vocab.json')           # shared 50-word list (in git)
HAND_MODEL = os.path.join(MODELS_DIR, 'hand_landmarker.task')
POSE_MODEL = os.path.join(MODELS_DIR, 'yolo11n-pose.pt')


def _migrate_old_layout():
    """older versions kept everything in the project folder itself: move it into data/ and models/"""
    moves = [('sign_data', DATA_DIR), ('sign_model.pt', MODEL_PATH), ('words.json', WORDS_PATH),
             ('settings.json', SETTINGS_PATH), ('train_history.json', HISTORY_PATH),
             ('hand_landmarker.task', HAND_MODEL), ('yolo11n-pose.pt', POSE_MODEL)]
    for old, new in moves:
        src = os.path.join(ROOT, old)
        if not os.path.exists(src):
            continue
        if os.path.isdir(new) and not os.listdir(new):
            os.rmdir(new)                      # empty placeholder folder: replace it
        if os.path.exists(new):
            print(f'kept {old} in place: {os.path.relpath(new, ROOT)} already exists')
            continue
        os.makedirs(os.path.dirname(new), exist_ok=True)
        shutil.move(src, new)
        print(f'moved {old} -> {os.path.relpath(new, ROOT)}')


os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
_migrate_old_layout()
os.makedirs(DATA_DIR, exist_ok=True)
os.chdir(MODELS_DIR)          # YOLO downloads its model file into the current folder
HAND_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'
WORD_ID_RE = re.compile(r'^[a-z0-9_]{1,32}$')
# own clip: <time ms>.npz   merged clip from another person: <contributor ID>__<content hash>.npz
CLIP_RE = re.compile(r'^([0-9]{6,20}\.(npz|npy)|[a-z0-9-]{2,24}__[0-9a-f]{16}\.npz)$')
CONTRIB_RE = re.compile(r'^[a-z0-9-]{2,24}$')

# ---------------------------------------------------------------- Windows: keep full speed when the window is covered
def keep_full_speed():
    """Windows 11 slows down a program whose window is hidden behind others (e.g. the browser maximized over the
    black window): power-saving 'EcoQoS' cores and coarse timers. Opt this process out of that."""
    if os.name != 'nt':
        return 'not Windows'
    import ctypes
    from ctypes import wintypes
    done = []
    try:
        class PowerThrottling(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        # ProcessPowerThrottling = 4; EXECUTION_SPEED 0x1 | IGNORE_TIMER_RESOLUTION 0x4 controlled, state 0 = off
        st = PowerThrottling(1, 0x1 | 0x4, 0)
        if k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)):
            done.append('power throttling off')
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        if k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000):     # ABOVE_NORMAL_PRIORITY_CLASS
            done.append('priority above normal')
        if ctypes.WinDLL('winmm').timeBeginPeriod(1) == 0:
            done.append('timer 1 ms')
    except Exception as e:  # noqa: BLE001
        done.append(f'failed: {e}')
    return ', '.join(done) or 'no change'


# ---------------------------------------------------------------- settings
DEFAULT_SETTINGS = {
    'window_sec': 4.0,     # length the model looks at (longest sign must fit)
    'fps': 12,             # frames per second stored in clips / fed to the model
    'thresh': 0.75,        # live: confidence needed
    'need_frames': 3,      # live: consecutive frames above thresh
    'min_hand': 0.3,       # live: hands visible in at least this share of the window
    'hold_sec': 3.0,       # live: how long a recognized word stays on screen
    'cooldown': 2.0,       # live: no second detection within this time
    'ready_sec': 1.5,      # record: countdown before each clip
    'gap_sec': 1.0,        # record: pause between auto clips
    'epochs': 300,         # train
    'person_conf': 0.3,    # tracker: person detection confidence
    'wrist_match': 0.6,    # tracker: hand must be this close (x shoulder width) to the signer's wrist
    'cam_width': 1280,     # camera: requested resolution / frame rate (MJPG); the camera picks the closest it supports
    'cam_height': 720,
    'cam_fps': 30,
}
SETTING_LIMITS = {
    'window_sec': (1.0, 10.0), 'fps': (5, 30), 'thresh': (0.3, 0.99), 'need_frames': (1, 20),
    'min_hand': (0.0, 1.0), 'hold_sec': (0.5, 10.0), 'cooldown': (0.0, 10.0), 'ready_sec': (0.0, 5.0),
    'gap_sec': (0.0, 5.0), 'epochs': (50, 2000), 'person_conf': (0.1, 0.9), 'wrist_match': (0.2, 2.0),
    'cam_width': (320, 3840), 'cam_height': (240, 2160), 'cam_fps': (5, 120),
}


def _read_json(path, default):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    s.update({k: v for k, v in _read_json(SETTINGS_PATH, {}).items() if k in DEFAULT_SETTINGS})
    return s


def save_settings(new):
    s = load_settings()
    for k, v in new.items():
        if k not in DEFAULT_SETTINGS:
            continue
        lo, hi = SETTING_LIMITS[k]
        v = float(v)
        v = min(max(v, lo), hi)
        s[k] = int(round(v)) if isinstance(DEFAULT_SETTINGS[k], int) else v
    _write_json(SETTINGS_PATH, s)
    return s


def seq_len(settings=None):
    s = settings or load_settings()
    return int(round(s['window_sec'] * s['fps']))


# ---------------------------------------------------------------- words
DEFAULT_WORDS = [
    {'id': 'none', 'name': '(동작 없음)', 'record_sec': 3.0},
    {'id': 'hungry', 'name': '배고파', 'record_sec': 3.0},
    {'id': 'thirsty', 'name': '목말라', 'record_sec': 4.0},
]


def load_words():
    words = _read_json(WORDS_PATH, None)
    if not isinstance(words, list):
        words = [dict(w) for w in DEFAULT_WORDS]
        # adopt any data folder that is not listed yet
        if os.path.isdir(DATA_DIR):
            known = {w['id'] for w in words}
            for d in sorted(os.listdir(DATA_DIR)):
                if WORD_ID_RE.match(d) and d not in known and os.path.isdir(os.path.join(DATA_DIR, d)):
                    words.append({'id': d, 'name': d, 'record_sec': 3.0})
        _write_json(WORDS_PATH, words)
    if not any(w['id'] == 'none' for w in words):
        words.insert(0, dict(DEFAULT_WORDS[0]))
        _write_json(WORDS_PATH, words)
    return words


def save_words(words):
    _write_json(WORDS_PATH, words)


def get_word(word_id):
    return next((w for w in load_words() if w['id'] == word_id), None)


def load_vocab():
    """the shared 50-word list (app/vocab.json). Their IDs are fixed and reserved."""
    try:
        with open(VOCAB_PATH, encoding='utf-8') as f:
            return json.load(f)['words']
    except (OSError, ValueError, KeyError):
        return []


def vocab_alias():
    """shared word ID or one of its aliases (e.g. cups) -> shared word ID"""
    out = {}
    for v in load_vocab():
        out[v['id']] = v['id']
        for a in v.get('aliases', []):
            out[a] = v['id']
    return out


def add_word(word_id, name, record_sec, shared=False, extra=None):
    """shared=False: a personal word - its ID must not be one of the shared 50 (or their aliases)"""
    word_id = (word_id or '').strip().lower()
    name = (name or '').strip()
    if not WORD_ID_RE.match(word_id):
        raise ValueError('영문 ID는 영어 소문자, 숫자, _ 만 쓸 수 있어요 (최대 32자).')
    if word_id == 'none':
        raise ValueError('none 은 (동작 없음) 전용 ID예요.')
    if not shared and word_id in vocab_alias():
        raise ValueError(f'"{word_id}" 는 공유 단어 50개에 쓰는 ID예요. 위 목록에서 골라 추가하거나 다른 ID를 써 주세요.')
    if not name or len(name) > 30:
        raise ValueError('화면에 표시할 이름을 1~30자로 입력해 주세요.')
    words = load_words()
    if any(w['id'] == word_id for w in words):
        raise ValueError(f'이미 있는 ID예요: {word_id}')
    sec = min(max(float(record_sec), 0.5), 10.0)
    words.append({'id': word_id, 'name': name, 'record_sec': sec, **(extra or {})})
    save_words(words)
    os.makedirs(os.path.join(DATA_DIR, word_id), exist_ok=True)
    return words


def add_vocab_word(vocab_id):
    """adds one of the shared 50 words with its fixed ID, name and record length"""
    v = next((v for v in load_vocab() if v['id'] == vocab_id), None)
    if v is None:
        raise ValueError('50개 목록에 없는 단어예요.')
    alias = vocab_alias()
    if any(alias.get(w['id']) == vocab_id for w in load_words()):
        raise ValueError(f'이미 추가한 단어예요: {v["name"]}')
    return add_word(v['id'], v['name'], v.get('record_sec', 3.0), shared=True)


def update_word(word_id, name=None, record_sec=None):
    words = load_words()
    w = next((w for w in words if w['id'] == word_id), None)
    if w is None:
        raise ValueError('없는 단어예요.')
    if name is not None:
        name = name.strip()
        if not name or len(name) > 30:
            raise ValueError('이름을 1~30자로 입력해 주세요.')
        w['name'] = name
    if record_sec is not None:
        w['record_sec'] = min(max(float(record_sec), 0.5), 10.0)
    save_words(words)
    return words


def delete_word(word_id):
    """removes the word and deletes its recorded clips right away (cannot be undone)"""
    if word_id == 'none':
        raise ValueError('"동작 없음"은 지울 수 없어요.')
    words = load_words()
    if not any(w['id'] == word_id for w in words):
        raise ValueError('없는 단어예요.')
    src = os.path.join(DATA_DIR, word_id)
    if os.path.isdir(src):
        shutil.rmtree(src)
    save_words([w for w in words if w['id'] != word_id])


def reset_dataset():
    """전체 초기화: deletes every clip, the word list (back to the defaults), the trained model, the training history
    and the deleted-clip list right away (cannot be undone). Keeps the temporary ID and the settings."""
    n = sum(len(fs) for _, _, fs in os.walk(DATA_DIR))
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    for p in (WORDS_PATH, MODEL_PATH, HISTORY_PATH, DELETED_PATH):
        if os.path.exists(p):
            os.remove(p)
    load_words()                       # writes the default word list again
    return n


def labels_ko():
    return {w['id']: w['name'] for w in load_words()}


# ---------------------------------------------------------------- clips
POSE_IDX = [0, 5, 6, 7, 8, 9, 10]   # nose, shoulders, elbows, wrists
HAND_BASE = len(POSE_IDX) * 2
FEAT_DIM = HAND_BASE + 21 * 2 * 2 + 2


def resample(seq, n):
    seq = np.asarray(seq, np.float32)
    if len(seq) == 0:
        return np.zeros((n, FEAT_DIM), np.float32)
    idx = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (idx - lo)[:, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)


def clip_files(word_id):
    d = os.path.join(DATA_DIR, word_id)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if CLIP_RE.match(f))


def clip_contributor(fname):
    """None for your own clips, else the contributor ID of a merged clip"""
    return fname.split('__', 1)[0] if '__' in fname else None


def load_clip(word_id, fname):
    """returns (seq (T, FEAT_DIM), seconds)"""
    path = os.path.join(DATA_DIR, word_id, fname)
    if fname.endswith('.npz'):
        with np.load(path) as d:
            return d['seq'].astype(np.float32), float(d['sec'])
    seq = np.load(path).astype(np.float32)
    sec = 2.0 if len(seq) == 30 else len(seq) / 12.0   # files from the old scripts
    return seq, sec


PAD_SEC = 1.0     # recording keeps this much extra before and after the red-frame part


def save_clip(word_id, frames, sec, fps, pre=0.0, post=0.0):
    """frames cover pre + sec + post seconds. `core` = where the red-frame part is (seconds from the start)."""
    d = os.path.join(DATA_DIR, word_id)
    os.makedirs(d, exist_ok=True)
    total = pre + sec + post
    n = max(4, int(round(total * fps)))
    fname = f'{int(time.time() * 1000)}.npz'
    extra = {'core': np.array([pre, pre + sec], np.float32)} if (pre > 0 or post > 0) else {}
    np.savez(os.path.join(d, fname), seq=resample(frames, n), sec=np.float32(total), **extra)
    return fname


def load_clip_core(word_id, fname):
    """(seq, seconds, core) - core = (start, end) seconds of the red-frame part, None for clips without padding"""
    path = os.path.join(DATA_DIR, word_id, fname)
    if fname.endswith('.npz'):
        with np.load(path) as d:
            core = tuple(float(x) for x in d['core']) if 'core' in d else None
            return d['seq'].astype(np.float32), float(d['sec']), core
    seq, sec = load_clip(word_id, fname)
    return seq, sec, None


def _activity(seq, fps):
    """per frame: how fast the hands / wrists move into this frame (shoulder widths per second)"""
    n = len(seq)
    sp = np.zeros(n, np.float32)
    if n < 2:
        return sp
    for s in (0, 1):
        pts = seq[:, HAND_BASE + s * 42: HAND_BASE + s * 42 + 42].reshape(n, 21, 2)
        here = seq[:, HAND_BASE + 84 + s] >= 0.5
        palm = pts[:, [0, 5, 9, 13, 17]].mean(1)
        d = np.linalg.norm(palm[1:] - palm[:-1], axis=1)
        sp[1:] = np.maximum(sp[1:], np.where(here[1:] & here[:-1], d, 0))
    for k in (5, 6):                                   # wrists from the body pose
        p = seq[:, 2 * k: 2 * k + 2]
        ok = (p != 0).any(1)
        d = np.linalg.norm(p[1:] - p[:-1], axis=1)
        sp[1:] = np.maximum(sp[1:], np.where(ok[1:] & ok[:-1], d, 0))
    return sp * fps


MOVE_SPEED = 0.5      # faster than this (shoulder widths / s) counts as part of the sign


def train_part(seq, sec, core):
    """the part of a clip that is used for training (and shown in 수어 보기):
    the red-frame part, widened into the extra 1 s before / after only while the hands are still moving."""
    if core is None or len(seq) < 4:
        return seq, sec
    n = len(seq)
    fps = n / max(sec, 1e-6)
    i0 = int(round(core[0] * fps))
    i1 = int(round(core[1] * fps))
    i0, i1 = max(0, min(i0, n - 2)), max(i0 + 2, min(i1, n))
    act = _activity(seq, fps)
    while i0 > 0 and act[i0] > MOVE_SPEED:            # moving into the first frame: the sign started earlier
        i0 -= 1
    while i1 < n and act[i1] > MOVE_SPEED:            # still moving after the last frame: it ended later
        i1 += 1
    return seq[i0:i1], (i1 - i0) / fps


def load_train_clip(word_id, fname):
    """(seq, seconds) of the part used for training"""
    return train_part(*load_clip_core(word_id, fname))


def delete_clips(word_id, fnames):
    """deletes clip files right away (cannot be undone); returns how many were deleted"""
    deleted = 0
    for f in fnames:
        src = os.path.join(DATA_DIR, word_id, f)
        if CLIP_RE.match(f) and os.path.isfile(src):
            os.remove(src)
            deleted += 1
    return deleted


def hand_ratio(seq):
    seq = np.asarray(seq)
    if len(seq) == 0:
        return 0.0
    return float((seq[:, HAND_BASE + 84:HAND_BASE + 86].max(1) > 0.5).mean())


def fit_window(seq, sec, window_sec, L):
    """place a clip of `sec` seconds into the model window (window_sec, L frames).
    shorter clips sit in the middle with still frames around; longer clips are squeezed."""
    seq = np.asarray(seq, np.float32)
    if sec >= window_sec * 0.98:
        return resample(seq, L)
    n = max(2, int(round(L * sec / window_sec)))
    sq = resample(seq, n)
    off = (L - n) // 2
    return np.concatenate([np.repeat(sq[:1], off, 0), sq, np.repeat(sq[-1:], L - n - off, 0)]).astype(np.float32)


# ---------------------------------------------------------------- features
KP_CONF = 0.5
FINGERS = {
    'Thumb':  [1, 2, 3, 4],
    'Index':  [5, 6, 7, 8],
    'Middle': [9, 10, 11, 12],
    'Ring':   [13, 14, 15, 16],
    'Pinky':  [17, 18, 19, 20],
}
COLORS = {'Thumb': (60, 60, 255), 'Index': (0, 165, 255), 'Middle': (0, 230, 255),
          'Ring': (80, 220, 80), 'Pinky': (255, 120, 60)}
PALM = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
ARMS = {'Left': (5, 7, 9), 'Right': (6, 8, 10)}
ARM_COLORS = {'Left': (255, 0, 255), 'Right': (255, 255, 0)}
PER_HAND_SHAPE = 42 + 15 + 4     # wrist-relative points + joint angles + thumb-tip distances
SHAPE_DIM = 2 * PER_HAND_SHAPE
MOTION_DIM = 2 * 5 + 2 * 3       # per hand: palm velocity(2) + wrist velocity(2) + palm speed; per arm wrist: velocity(2) + speed
MOTION_WEIGHT = 4.0              # movement (e.g. drawing a circle) must count as much as hand shape
FULL_DIM = FEAT_DIM + SHAPE_DIM + MOTION_DIM
SHAPE_WEIGHT = 2.0               # make finger shape count more than arm position
CHAINS = [[0] + FINGERS[n] for n in FINGERS]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(p):
    s = {'Thumb': dist(p[4], p[9]) > dist(p[3], p[9]) and dist(p[4], p[17]) > dist(p[2], p[17])}
    for name in ['Index', 'Middle', 'Ring', 'Pinky']:
        idx = FINGERS[name]
        s[name] = dist(p[idx[3]], p[0]) > dist(p[idx[1]], p[0]) * 1.1
    return s


def build_features(kp, cf, box, hands):
    """one frame -> FEAT_DIM vector, normalized to the signer's shoulders"""
    if cf[5] > KP_CONF and cf[6] > KP_CONF:
        center = (kp[5] + kp[6]) / 2
        scale = float(np.linalg.norm(kp[5] - kp[6]))
    else:
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
        scale = float(box[3] - box[1]) / 4
    scale = max(scale, 1.0)
    f = np.zeros(FEAT_DIM, np.float32)
    for i, j in enumerate(POSE_IDX):
        if cf[j] > KP_CONF:
            f[i * 2:i * 2 + 2] = (kp[j] - center) / scale
    slots = {}
    for side, pts in hands:
        s = 0 if side == 'Left' else 1
        if s in slots:
            s = 1 - s
        if s in slots:
            continue
        slots[s] = pts
    for s, pts in slots.items():
        arr = (np.array(pts, np.float32) - center) / scale
        f[HAND_BASE + s * 42: HAND_BASE + s * 42 + 42] = arr.reshape(-1)
        f[HAND_BASE + 84 + s] = 1.0
    return f


def _joint_angles(p, chain):
    out = []
    for k in range(1, 4):
        a = p[:, chain[k - 1]] - p[:, chain[k]]
        b = p[:, chain[k + 1]] - p[:, chain[k]]
        cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-6)
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))) / 180.0)
    return np.stack(out, -1)


def _velocity(pos, present):
    """frame-to-frame movement; 0 where the point is missing in either frame; clipped against tracking jumps"""
    T = len(pos)
    v = np.zeros((T, 2), np.float32)
    if T > 1:
        v[1:] = pos[1:] - pos[:-1]
        ok = np.zeros(T, bool)
        ok[1:] = present[1:] & present[:-1]
        v[~ok] = 0
    return np.clip(v, -0.5, 0.5)


def motion_features(seq):
    """(T, FEAT_DIM) -> (T, MOTION_DIM): how the hands and wrists move (direction + speed)"""
    seq = np.asarray(seq, np.float32)
    T = seq.shape[0]
    out = np.zeros((T, MOTION_DIM), np.float32)
    o = 0
    for s in range(2):
        present = seq[:, HAND_BASE + 84 + s] > 0.5
        p = seq[:, HAND_BASE + s * 42: HAND_BASE + s * 42 + 42].reshape(T, 21, 2)
        vp = _velocity(p[:, [0, 5, 9, 13, 17]].mean(1), present)     # palm centre
        vw = _velocity(p[:, 0], present)                              # wrist
        out[:, o:o + 2], out[:, o + 2:o + 4] = vp, vw
        out[:, o + 4] = np.linalg.norm(vp, axis=1)
        o += 5
    for i in (5, 6):                                                  # arm wrists from the body pose
        w = seq[:, 2 * i:2 * i + 2]
        v = _velocity(w, np.any(w != 0, axis=1))
        out[:, o:o + 2] = v
        out[:, o + 2] = np.linalg.norm(v, axis=1)
        o += 3
    return out * MOTION_WEIGHT


def add_shape_features(seq):
    """(T, FEAT_DIM) raw features -> (T, FULL_DIM): + hand-size-normalized finger shape + movement"""
    seq = np.asarray(seq, np.float32)
    T = seq.shape[0]
    extra = np.zeros((T, SHAPE_DIM), np.float32)
    for s in range(2):
        present = seq[:, HAND_BASE + 84 + s] > 0.5
        if not present.any():
            continue
        p = seq[:, HAND_BASE + s * 42: HAND_BASE + s * 42 + 42].reshape(T, 21, 2)
        size = np.maximum(np.linalg.norm(p[:, 9] - p[:, 0], axis=-1), 1e-3)
        local = (p - p[:, :1]) / size[:, None, None]
        ang = np.concatenate([_joint_angles(p, c) for c in CHAINS], -1)
        tips = np.stack([np.linalg.norm(p[:, 4] - p[:, t], axis=-1) for t in (8, 12, 16, 20)], -1) / size[:, None]
        feat = np.concatenate([local.reshape(T, 42), ang, tips], -1) * SHAPE_WEIGHT
        feat[~present] = 0
        extra[:, s * PER_HAND_SHAPE:(s + 1) * PER_HAND_SHAPE] = feat
    return np.concatenate([seq, extra, motion_features(seq)], -1).astype(np.float32)


# ---------------------------------------------------------------- model
class SignNet(torch.nn.Module):
    def __init__(self, n_cls, hidden=96):
        super().__init__()
        self.gru = torch.nn.GRU(FULL_DIM, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden, n_cls)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


class Recognizer:
    """sliding-window live recognition using the saved model.
    add() is cheap and is called for every camera frame; step() runs the model and is called
    about 10 times a second (from its own thread in the web app), so the camera never slows down."""

    STEP_SEC = 0.1     # how often the model is run (10 decisions per second)

    def __init__(self, device=None):
        import threading
        self._lk = threading.Lock()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.labels = []
        self.error = None
        self.window_sec, self.L = 4.0, 48
        self._last_step = 0.0
        self.reset()
        self.reload()

    def reset(self):
        with self._lk:
            self.buf = collections.deque()
            self.probs = None
            self.hand_ratio = 0.0
            self.streak = 0
            self.current = None
            self.current_until = 0.0
            self.cooldown_until = 0.0
            self.peaks = []
            self.history = []

    def reload(self):
        model, labels, error, people = None, [], None, {}
        window_sec, L = 4.0, 48
        if not os.path.exists(MODEL_PATH):
            error = '학습된 모델이 없어요. 먼저 학습을 실행해 주세요.'
        else:
            try:
                ck = torch.load(MODEL_PATH, map_location=self.device)
                if ck.get('dim') != FULL_DIM:
                    raise ValueError('old format')
                model = SignNet(len(ck['labels']), ck.get('hidden', 96)).to(self.device)
                model.load_state_dict(ck['state'])
                model.eval()
                labels = list(ck['labels'])
                window_sec, L = float(ck.get('window_sec', 4.0)), int(ck.get('seq_len', 48))
                people = ck.get('contributors') or {}      # word -> {contributor ID or '나': clips} (newer models)
            except Exception as e:  # noqa: BLE001
                model, error = None, f'모델을 불러오지 못했어요. 다시 학습해 주세요. ({e})'
        with self._lk:
            self.model, self.labels, self.error, self.people = model, labels, error, people
            self.window_sec, self.L = window_sec, L
        self.reset()

    def add(self, now, feat):
        """store one camera frame (cheap)"""
        with self._lk:
            if self.model is None:
                return
            self.buf.append((now, feat if feat is not None else np.zeros(FEAT_DIM, np.float32)))
            while self.buf and now - self.buf[0][0] > self.window_sec:
                self.buf.popleft()

    def step(self, now, s):
        """run the model on the current window; s = settings. returns a newly recognized word id or None"""
        with self._lk:
            model = self.model
            if model is None or len(self.buf) < 8 or now - self.buf[0][0] <= self.window_sec * 0.8:
                return None
            frames = [f for _, f in self.buf]
            labels, L = self.labels, self.L
        win = resample(frames, L)                      # heavy part runs without holding the lock
        hr = hand_ratio(win)
        x = torch.tensor(add_shape_features(win), dtype=torch.float32)[None].to(self.device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        k = int(np.argmax(probs))
        lab = labels[k]
        best = max((float(p) for l, p in zip(labels, probs) if l != 'none'), default=0.0)
        hit = None
        with self._lk:
            if self.model is not model:                # model was reloaded meanwhile
                return None
            self.probs, self.hand_ratio = probs, hr
            self.peaks = [(t, p) for t, p in self.peaks if now - t <= 3.0] + [(now, best)]
            ok = lab != 'none' and probs[k] >= s['thresh'] and hr >= s['min_hand']
            self.streak = self.streak + 1 if ok else 0
            if self.streak >= s['need_frames'] and now >= self.cooldown_until:
                self.current, self.current_until = lab, now + s['hold_sec']
                self.cooldown_until = now + s['cooldown']
                self.history.insert(0, {'time': time.strftime('%H:%M:%S'), 'word': lab, 'prob': float(probs[k])})
                del self.history[20:]
                self.streak = 0
                hit = lab
        return hit

    def push(self, now, feat, s):
        """add a frame and run the model at most every STEP_SEC (used by sign_live.py)"""
        self.add(now, feat)
        if now - self._last_step >= self.STEP_SEC:
            self._last_step = now
            return self.step(now, s)
        return None

    def showing(self, now):
        return self.current if (self.current and now < self.current_until) else None

    def peak(self):
        return max((p for _, p in self.peaks), default=0.0)


# ---------------------------------------------------------------- drawing
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def box_int(b, w, h):
    return (max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3])))


def gray_out(img, others, keep):
    if not others:
        return
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for b in others:
        x1, y1, x2, y2 = box_int(b, w, h)
        mask[y1:y2, x1:x2] = True
    if keep is not None:
        x1, y1, x2, y2 = box_int(keep, w, h)
        mask[y1:y2, x1:x2] = False
    g = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 0.5).astype(np.uint8)
    img[mask] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)[mask]


def draw_hand(img, p, st):
    for a, b in PALM:
        cv2.line(img, p[a], p[b], (200, 200, 200), 2)
    for name, idx in FINGERS.items():
        up = st[name]
        col = COLORS[name] if up else (110, 110, 110)
        for a, b in zip(idx[:-1], idx[1:]):
            cv2.line(img, p[a], p[b], col, 4 if up else 2)
        for j in idx:
            cv2.circle(img, p[j], 4, col, -1)
    cv2.circle(img, p[0], 6, (255, 255, 255), -1)


def put(img, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


_fonts = {}


def font(size):
    if size not in _fonts:
        try:
            _fonts[size] = ImageFont.truetype(FONT_PATH, size)
        except OSError:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


PANEL_W = 400


def add_panel(frame, title, big, big_color, sub_lines):
    """fixed info panel on the right side (Korean text via PIL); colors are RGB"""
    h = frame.shape[0]
    panel = Image.new('RGB', (PANEL_W, h), (25, 25, 32))
    d = ImageDraw.Draw(panel)
    d.text((20, 18), title, font=font(28), fill=(220, 220, 220))
    d.line([(20, 62), (PANEL_W - 20, 62)], fill=(80, 80, 90), width=2)
    if big:
        d.text((20, 78), big, font=font(72), fill=big_color)
    y = 190
    for text, col in sub_lines:
        d.text((20, y), text, font=font(24), fill=col)
        y += 34
    panel = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    return np.hstack([frame, panel])


def _measure_fps(cap, first_sec=6.0, measure_frames=15, measure_sec=2.0):
    """wait for the first frame (Media Foundation can take a few seconds to start),
    then return the real frame rate of the following frames (0 if no frames arrive)"""
    t_end = time.time() + first_sec
    while time.time() < t_end:
        ok, _ = cap.read()
        if ok:
            break
    else:
        return 0.0
    for _ in range(3):                      # skip a few warm-up frames
        cap.read()
    n, t0, t_end = 0, time.time(), time.time() + measure_sec
    while n < measure_frames and time.time() < t_end:
        ok, _ = cap.read()
        if ok:
            n += 1
    return n / max(time.time() - t0, 1e-6)


def _try_camera(index, api, width, height, fps, mjpg=True):
    cap = cv2.VideoCapture(index, api) if api is not None else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None, 0.0
    cc = cv2.VideoWriter_fourcc(*'MJPG')
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cc)        # some drivers reset the format after a size change
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, _measure_fps(cap)


def _try_camera_timeout(timeout, *args, **kw):
    """Media Foundation can hang forever when the webcam is busy: give each try a time limit.
    returns (cap, fps, 'ok' | 'timeout' | error text)"""
    import threading
    box = {}

    def work():
        try:
            box['r'] = _try_camera(*args, **kw)
        except Exception as e:  # noqa: BLE001
            box['e'] = str(e)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, 0.0, 'timeout'
    if 'e' in box:
        return None, 0.0, box['e']
    cap, fps = box['r']
    return cap, fps, 'ok'


CAMERA_STATE = {'measured': 0.0, 'backend': ''}


def open_camera(index=0, width=None, height=None, fps=None):
    """open the webcam at its full frame rate.
    On Windows, Media Foundation (MSMF) is used first: it delivers compressed video (30 fps on a C920).
    DirectShow often ignores the MJPG request and falls back to uncompressed YUY2 (about 10 fps over USB),
    so it is only used when Media Foundation delivers no frames at all."""
    if width is None or height is None or fps is None:
        s = load_settings()
        width, height, fps = width or s['cam_width'], height or s['cam_height'], fps or s['cam_fps']
    if os.name == 'nt':
        tries = [('MSMF', cv2.CAP_MSMF, True), ('MSMF (camera default format)', cv2.CAP_MSMF, False),
                 ('DSHOW', cv2.CAP_DSHOW, True)]
    else:
        tries = [('default', None, True)]
    chosen, hung = None, False
    for name, api, mjpg in tries:
        print(f'Camera: trying {name} ...', flush=True)
        cap, measured, status = _try_camera_timeout(40.0, index, api, width, height, fps, mjpg=mjpg)
        if status == 'timeout':
            hung = True
            print(f'Camera via {name}: no answer within 40 s (webcam busy?)', flush=True)
            continue
        if cap is None:
            print(f'Camera via {name}: could not open ({status})', flush=True)
            continue
        print(f'Camera via {name}: {camera_info(cap)}, measured {measured:.1f} fps', flush=True)
        if measured > 0:
            # Media Foundation delivers frames -> use it, even if a bit slow; DirectShow only as last resort
            chosen = (cap, name, measured)
            break
        cap.release()                          # no frames: close (a webcam can be open only once) and try next
        time.sleep(0.5)
    if chosen is None:
        if hung:
            raise SystemExit('웹캠이 응답하지 않아요. 다른 프로그램(이전에 켠 서버, 카메라 앱, 화상회의 등)이 웹캠을 쓰고 있는지 '
                             '확인하고 닫은 뒤 1_web.bat을 다시 실행해 주세요. 그래도 안 되면 웹캠 USB를 뺐다 꽂아 주세요.')
        raise SystemExit('웹캠을 열지 못했어요. 웹캠이 연결돼 있는지 확인해 주세요.')
    cap, name, measured = chosen
    CAMERA_STATE.update(measured=measured, backend=name)
    print(f'Camera: using {name} - {camera_info(cap)}')
    return cap


def camera_info(cap):
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = ''.join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)).strip('\x00 ')
    if len(fourcc) != 4 or not fourcc.isalnum():   # Media Foundation reports a numeric format code
        fourcc = 'compressed' if cc else '?'
    try:
        backend = cap.getBackendName()
    except Exception:  # noqa: BLE001
        backend = ''
    return f'{w}x{h} @ {fps:.0f} fps ({fourcc}{", " + backend if backend else ""})'


class CameraReader:
    """reads the webcam in its own thread and always hands out the newest frame,
    so slow processing never makes the camera fall behind"""

    def __init__(self, cap):
        import threading
        self.cap = cap
        self.cond = threading.Condition()
        self.frame, self.n, self.ok, self.stopped = None, 0, True, False
        self.cam_fps = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        prev = time.time()
        while not self.stopped:
            ok, frame = self.cap.read()
            now = time.time()
            with self.cond:
                self.ok = ok
                if ok:
                    self.frame, self.n = frame, self.n + 1
                    self.cam_fps = 0.9 * self.cam_fps + 0.1 / max(now - prev, 1e-6)
                    prev = now
                self.cond.notify_all()
            if not ok:
                time.sleep(0.02)

    def read(self, last_n, timeout=1.0):
        """wait for a frame newer than last_n; returns (ok, frame, n)"""
        with self.cond:
            self.cond.wait_for(lambda: self.n != last_n or self.stopped, timeout=timeout)
            if self.n == last_n or self.frame is None:
                return False, None, last_n
            return True, self.frame, self.n

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()


# ---------------------------------------------------------------- tracker
def _inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


class SignerTracker:
    """YOLO pose (closest person = signer) + MediaPipe hands for the signer only"""

    def __init__(self, person_conf=0.3, switch_ratio=1.3, wrist_match=0.6):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from ultralytics import YOLO
        self.mp = mp
        if not os.path.exists(HAND_MODEL):
            print('Downloading hand model...')
            urllib.request.urlretrieve(HAND_URL, HAND_MODEL)
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print('Pose device:', torch.cuda.get_device_name(0) if self.device == 0 else 'CPU')
        self.pose = YOLO(POSE_MODEL if os.path.exists(POSE_MODEL) else 'yolo11n-pose.pt')  # downloads into models/
        self.person_conf = person_conf
        self.switch_ratio = switch_ratio
        self.wrist_match = wrist_match
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.prev_box = None
        self.t0 = time.time()
        self.last_ts = -1

    def _is_signer_hand(self, p, kp, cf, box, others):
        wx, wy = p[0]
        if cf[5] > KP_CONF and cf[6] > KP_CONF:
            sw = float(np.linalg.norm(kp[5] - kp[6]))
        else:
            sw = float(box[3] - box[1]) / 4
        sw = max(sw, 20.0)
        for j in (9, 10):  # signer's wrists from YOLO pose
            if cf[j] > 0.3 and math.hypot(wx - kp[j][0], wy - kp[j][1]) < self.wrist_match * sw:
                return True
        # wrist not visible: accept only if the hand is inside the signer box and not in anyone else's
        return _inside(p[0], box) and not any(_inside(p[0], b) for b in others)

    def process(self, frame):
        """frame: mirrored BGR image (drawn on in place).
        returns (frame, feature vector or None, info lines, n_hands)"""
        clean = frame.copy()
        h, w = frame.shape[:2]
        lines = []
        r = self.pose.predict(clean, device=self.device, conf=self.person_conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            self.prev_box = None
            lines.append(('No person', (0, 0, 255)))
            return frame, None, lines, 0
        boxes = r.boxes.xyxy.cpu().numpy()
        kxy = r.keypoints.xy.cpu().numpy()
        kcf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best = int(np.argmax(areas))  # biggest box = closest person
        if self.prev_box is not None:
            ious = [iou(b, self.prev_box) for b in boxes]
            cur = int(np.argmax(ious))
            if ious[cur] > 0.3 and areas[best] < areas[cur] * self.switch_ratio:
                best = cur
        target = best
        self.prev_box = boxes[target]
        others = [b for i, b in enumerate(boxes) if i != target]
        gray_out(frame, others, boxes[target])
        for b in others:
            ox1, oy1, ox2, oy2 = box_int(b, w, h)
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (150, 150, 150), 2)
            cv2.putText(frame, 'other (ignored)', (ox1 + 4, oy1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        lines.append((f'People: {len(boxes)}  (signer = closest)', (0, 255, 255)))

        x1, y1, x2, y2 = boxes[target]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(frame, 'SIGNER', (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        kp = kxy[target]
        cf = kcf[target]
        kpi = [(int(x), int(y)) for x, y in kp]
        if cf[5] > KP_CONF and cf[6] > KP_CONF:
            cv2.line(frame, kpi[5], kpi[6], (200, 200, 200), 3)
        for side, (s, e, wr) in ARMS.items():
            col = ARM_COLORS[side]
            if cf[s] > KP_CONF and cf[e] > KP_CONF:
                cv2.line(frame, kpi[s], kpi[e], col, 6)
            if cf[e] > KP_CONF and cf[wr] > KP_CONF:
                cv2.line(frame, kpi[e], kpi[wr], col, 6)
            for j in (s, e, wr):
                if cf[j] > KP_CONF:
                    cv2.circle(frame, kpi[j], 8, col, -1)

        # image for the hand model: other people blacked out, signer box kept as is
        src = clean.copy()
        for b in others:
            ox1, oy1, ox2, oy2 = box_int(b, w, h)
            src[oy1:oy2, ox1:ox2] = 0
        sx1, sy1, sx2, sy2 = box_int(boxes[target], w, h)
        src[sy1:sy2, sx1:sx2] = clean[sy1:sy2, sx1:sx2]
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = int(max(0, x1 - 0.2 * bw)), int(max(0, y1 - 0.1 * bh))
        cx2, cy2 = int(min(w, x2 + 0.2 * bw)), int(min(h, y2 + 0.1 * bh))
        crop = src[cy1:cy2, cx1:cx2]
        hands = []
        rejected = 0
        if crop.size > 0:
            ch, cw = crop.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ts = max(int((time.time() - self.t0) * 1000), self.last_ts + 1)
            self.last_ts = ts
            res = self.landmarker.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)
            for i, lms in enumerate(res.hand_landmarks):
                p = [(int(l.x * cw) + cx1, int(l.y * ch) + cy1) for l in lms]
                if not self._is_signer_hand(p, kp, cf, boxes[target], others):
                    rejected += 1
                    continue
                hside = res.handedness[i][0].category_name
                draw_hand(frame, p, finger_states(p))
                hands.append((hside, p))
        txt = f'Hands: {len(hands)}' + (f'  (ignored {rejected})' if rejected else '')
        lines.append((txt, (0, 255, 0) if hands else (150, 150, 150)))
        return frame, build_features(kp, cf, boxes[target], hands), lines, len(hands)


# ---------------------------------------------------------------- compatibility for older scripts
LABELS_KO = labels_ko()
CLIP_SEC = load_settings()['window_sec']
SEQ_LEN = seq_len()
