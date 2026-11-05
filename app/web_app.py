# 수어 학습기 by Langpeu - (c) 2026 Langpeu, licensed under AGPL-3.0 (see LICENSE)
# Sign language trainer web app (local only): http://127.0.0.1:8000
# words / recording / training / live recognition / clip management / settings / history
import os, sys, json, time, threading, traceback, webbrowser
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS', '0')   # must be set BEFORE cv2 is imported: makes the webcam open in seconds instead of ~1 minute
import cv2
import numpy as np

import sign_common as sc
import sign_share

FAKE = os.environ.get('SIGN_FAKE_CAM') == '1'   # test mode without webcam / models
WEB_DIR = os.path.join(sc.ROOT, 'web')
HOST = '127.0.0.1'
EXIT_FROM_WEB = 99      # exit code that tells 1_web.bat to close its window too
LOW_HAND = 0.5          # clips with hands visible in less than this share are flagged


# ======================================================================= camera worker
class Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.jpeg = None
        self.frame_id = 0
        self.status = 'starting'
        self.error = None
        self.mode = 'preview'            # preview | record | live
        self.settings = sc.load_settings()
        self.tracker = None
        self.recog = None
        self.fps = 0.0
        self.n_hands = 0
        self.person = False
        self.rec = {'state': 'idle', 'word': None, 'auto': False, 't': 0.0, 'buf': [],
                    'session': 0, 'last': None, 'recent': []}
        self.words = {w['id']: w for w in sc.load_words()}   # cached; refreshed when words change
        self.cam_info = ''
        self.cam_fps = 0.0
        self.reopen = False
        self.reader = None

    def refresh_words(self):
        words = {w['id']: w for w in sc.load_words()}
        with self.lock:
            self.words = words

    def _rec_sec(self, word_id):
        w = self.words.get(word_id)
        return w['record_sec'] if w else 3.0

    # ---------- control (called from HTTP threads)
    def start_record(self, word_id, auto):
        self.refresh_words()
        if word_id not in self.words:
            raise ValueError('없는 단어예요.')
        with self.lock:
            self.mode = 'record'
            r = self.rec
            if r['word'] != word_id:
                r['session'] = 0
                r['recent'] = []
            r.update(word=word_id, auto=bool(auto), state='ready', t=time.time(), buf=[])

    def stop_record(self):
        with self.lock:
            self.rec.update(state='idle', auto=False, buf=[])
            if self.mode == 'record':
                self.mode = 'preview'

    def set_mode(self, mode):
        if mode not in ('preview', 'live'):
            raise ValueError('bad mode')
        with self.lock:
            if self.rec['state'] != 'idle':
                self.rec.update(state='idle', auto=False, buf=[])
            self.mode = mode
            if mode == 'live' and self.recog:
                self.recog.reset()

    def apply_settings(self, s):
        with self.lock:
            old = self.settings
            self.settings = s
            if any(old[k] != s[k] for k in ('cam_width', 'cam_height', 'cam_fps')):
                self.reopen = True          # camera loop reopens the webcam with the new mode
            if self.tracker:
                self.tracker.person_conf = s['person_conf']
                self.tracker.wrist_match = s['wrist_match']

    def forget_clips(self, word_id, files):
        with self.lock:
            r = self.rec
            if r['word'] == word_id:
                r['recent'] = [c for c in r['recent'] if c['file'] not in files]
                if r['last'] and r['last']['file'] in files:
                    r['last'] = r['recent'][0] if r['recent'] else None

    def reload_model(self):
        with self.lock:
            if self.recog:
                self.recog.reload()

    # ---------- loop
    def _placeholder(self, text):
        img = np.full((360, 640, 3), 30, np.uint8)
        cv2.putText(img, text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return img

    def _publish(self, img):
        h, w = img.shape[:2]
        if w > 960:
            img = cv2.resize(img, (960, int(h * 960 / w)))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.frame_id += 1
                self.cond.notify_all()

    def _fake_frame(self, t):
        img = np.full((720, 1280, 3), 40, np.uint8)
        x = int(640 + 300 * np.sin(t))
        cv2.circle(img, (x, 360), 60, (0, 200, 0), -1)
        cv2.putText(img, 'FAKE CAMERA (test mode)', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        f = np.zeros(sc.FEAT_DIM, np.float32)
        f[:sc.HAND_BASE] = np.random.randn(sc.HAND_BASE) * 0.1
        f[sc.HAND_BASE:sc.HAND_BASE + 84] = np.random.randn(84) * 0.3 + np.sin(t)
        f[sc.HAND_BASE + 84:] = 1.0
        return img, f

    def run(self):
        self._publish(self._placeholder('Starting camera and models...'))
        cap = None
        try:
            if not FAKE:
                self.tracker = sc.SignerTracker(person_conf=self.settings['person_conf'],
                                                wrist_match=self.settings['wrist_match'])
                self._open_camera()
            self.recog = sc.Recognizer()
            threading.Thread(target=self._infer_loop, daemon=True).start()
            self.status = 'running'
        except BaseException as e:  # noqa: BLE001  (open_camera raises SystemExit)
            self.status, self.error = 'error', f'카메라/모델 시작 실패: {e}'
            traceback.print_exc()
            self._publish(self._placeholder('Camera error - see the web page'))
            return
        prev, fails, last_n = time.time(), 0, -1
        while True:
            if FAKE:
                time.sleep(1 / 15)
                frame, feat = self._fake_frame(time.time())
                lines, nh = [('FAKE', (0, 255, 255))], 2
            else:
                if self.reopen:
                    self.reopen = False
                    try:
                        self.reader.release()
                        self._open_camera()
                        last_n = -1
                    except BaseException as e:  # noqa: BLE001
                        self.status, self.error = 'error', f'카메라를 다시 열지 못했어요: {e}'
                        time.sleep(1)
                        continue
                ok, frame, last_n = self.reader.read(last_n)
                self.cam_fps = self.reader.cam_fps
                if not ok:
                    fails += 1
                    if fails > 5:
                        self.status, self.error = 'error', '웹캠에서 영상을 받지 못했어요. 다른 프로그램이 카메라를 쓰고 있는지 확인해 주세요.'
                        self._publish(self._placeholder('No camera frames'))
                    continue
                if fails:
                    fails, self.status, self.error = 0, 'running', None
                frame = cv2.flip(frame, 1)
                frame, feat, lines, nh = self.tracker.process(frame)
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            self.n_hands, self.person = nh, feat is not None
            with self.lock:
                self._step_record(now, feat)
                live_on = self.mode == 'live'
                rec_on = self.rec['state'] == 'rec'
            if live_on:
                self.recog.add(now, feat)          # cheap; the model runs in _infer_loop
            for k, (text, c) in enumerate(lines):
                sc.put(frame, text, 30 + k * 30, c)
            if rec_on:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 14)
            self._publish(frame)

    def _infer_loop(self):
        """runs the sign model ~10 times a second, separately from the camera loop"""
        while True:
            t0 = time.time()
            if self.mode == 'live' and self.recog is not None:
                try:
                    self.recog.step(t0, self.settings)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
            time.sleep(max(0.0, sc.Recognizer.STEP_SEC - (time.time() - t0)))

    def _open_camera(self):
        s = self.settings
        cap = sc.open_camera(width=s['cam_width'], height=s['cam_height'], fps=s['cam_fps'])
        self.cam_info = sc.camera_info(cap)
        self.reader = sc.CameraReader(cap)

    def _step_record(self, now, feat):
        r, s = self.rec, self.settings
        if r['state'] == 'idle':
            return
        if r['state'] == 'ready' and now - r['t'] >= s['ready_sec']:
            r.update(state='rec', t=now, buf=[])
        if r['state'] == 'rec':
            r['buf'].append(feat if feat is not None else np.zeros(sc.FEAT_DIM, np.float32))
            sec = self._rec_sec(r['word'])
            if now - r['t'] >= sec:
                fname = sc.save_clip(r['word'], r['buf'], sec, s['fps'])
                seq, _ = sc.load_clip(r['word'], fname)
                hr = sc.hand_ratio(seq)
                remember_ratio(r['word'], fname, hr)
                r['session'] += 1
                r['last'] = {'file': fname, 'hand': hr, 'time': time.strftime('%H:%M:%S')}
                r['recent'].insert(0, r['last'])
                del r['recent'][15:]
                r.update(state='gap' if r['auto'] else 'idle', t=now, buf=[])
                if not r['auto']:
                    self.mode = 'preview'
        if r['state'] == 'gap' and now - r['t'] >= s['gap_sec']:
            r.update(state='ready', t=now)

    def snapshot(self):
        now = time.time()
        with self.lock:
            r, s = self.rec, self.settings
            remain = 0.0
            if r['state'] == 'ready':
                remain = s['ready_sec'] - (now - r['t'])
            elif r['state'] == 'rec':
                remain = self._rec_sec(r['word']) - (now - r['t'])
            elif r['state'] == 'gap':
                remain = s['gap_sec'] - (now - r['t'])
            live = {'model_ok': False, 'error': None}
            if self.recog:
                rg = self.recog
                live = {
                    'model_ok': rg.model is not None, 'error': rg.error, 'labels': rg.labels,
                    'probs': None if rg.probs is None else [round(float(p), 4) for p in rg.probs],
                    'showing': rg.showing(now), 'hand_ratio': rg.hand_ratio, 'peak': rg.peak(),
                    'history': rg.history[:10], 'thresh': s['thresh'], 'min_hand': s['min_hand'],
                }
            return {
                'status': self.status, 'error': self.error, 'mode': self.mode, 'fake': FAKE,
                'fps': round(self.fps, 1), 'cam_fps': round(self.cam_fps, 1), 'cam': self.cam_info,
                'hands': self.n_hands, 'person': self.person,
                'rec': {'state': r['state'], 'word': r['word'], 'auto': r['auto'],
                        'remain': max(0.0, round(remain, 1)), 'session': r['session'], 'last': r['last'],
                        'recent': list(r['recent'])},
                'live': live,
            }


W = Worker()

# ======================================================================= training job
TRAIN = {'running': False, 'progress': 0.0, 'log': [], 'result': None, 'error': None, 'stop': False}
TRAIN_LOCK = threading.Lock()


def _train_job():
    import sign_train

    def log(msg):
        print(msg)
        TRAIN['log'].append(msg)
        del TRAIN['log'][:-300]

    def progress(p):
        TRAIN['progress'] = float(p)

    try:
        TRAIN['result'] = sign_train.train_model(log=log, progress=progress, stop=lambda: TRAIN['stop'])
        W.reload_model()
    except sign_train.Stopped:
        TRAIN['error'] = '학습을 중지했어요. 이전 모델이 그대로 남아 있어요.'
        log('stopped')
    except Exception as e:  # noqa: BLE001
        TRAIN['error'] = str(e)
        log('ERROR: ' + str(e))
        traceback.print_exc()
    finally:
        TRAIN['running'] = False


def start_training():
    with TRAIN_LOCK:
        if TRAIN['running']:
            raise ValueError('이미 학습 중이에요.')
        TRAIN.update(running=True, progress=0.0, log=[], result=None, error=None, stop=False)
        threading.Thread(target=_train_job, daemon=True).start()


# ======================================================================= data views
_RATIO_CACHE = {}               # (word, file) -> hand ratio; clips never change after saving
_RATIO_LOCK = threading.Lock()


def remember_ratio(word_id, fname, ratio):
    with _RATIO_LOCK:
        _RATIO_CACHE[(word_id, fname)] = ratio


def clip_ratio(word_id, fname):
    with _RATIO_LOCK:
        r = _RATIO_CACHE.get((word_id, fname))
    if r is None:
        r = sc.hand_ratio(sc.load_clip(word_id, fname)[0])
        remember_ratio(word_id, fname, r)
    return r


def words_view():
    labels = set(W.recog.labels) if (W.recog and W.recog.model is not None) else set()
    alias = sc.vocab_alias()
    out = []
    for w in sc.load_words():
        files = sc.clip_files(w['id'])
        ratios = []
        for f in files:
            try:
                ratios.append(clip_ratio(w['id'], f))
            except Exception:  # noqa: BLE001
                pass
        out.append({**w, 'clips': len(files), 'hand': float(np.mean(ratios)) if ratios else None,
                    'low': int(sum(1 for x in ratios if x < LOW_HAND)),
                    'in_model': w['id'] in labels, 'shared': w['id'] == 'none' or w['id'] in alias})
    return out


def _clip_time(fname):
    try:
        return datetime.fromtimestamp(int(fname.split('.')[0]) / 1000).strftime('%m-%d %H:%M:%S')
    except (ValueError, OSError):
        return ''


def clips_view(word_id):
    if sc.get_word(word_id) is None:
        raise ValueError('없는 단어예요.')
    out = []
    for f in reversed(sc.clip_files(word_id)):
        try:
            seq, sec = sc.load_clip(word_id, f)
            who = sc.clip_contributor(f)
            t = _clip_time(f)
            if who:
                with np.load(os.path.join(sc.DATA_DIR, word_id, f)) as d:
                    rec = int(d['recorded']) if 'recorded' in d else 0
                t = datetime.fromtimestamp(rec / 1000).strftime('%m-%d %H:%M:%S') if rec else ''
            out.append({'file': f, 'sec': round(sec, 2), 'frames': len(seq), 'hand': sc.hand_ratio(seq),
                        'time': t, 'who': who})
        except Exception as e:  # noqa: BLE001
            out.append({'file': f, 'error': str(e), 'time': _clip_time(f)})
    return out


def clip_data(word_id, fname):
    if sc.get_word(word_id) is None or not sc.CLIP_RE.match(fname or ''):
        raise ValueError('잘못된 클립이에요.')
    seq, sec = sc.load_clip(word_id, fname)
    return {'sec': sec, 'hand_base': sc.HAND_BASE, 'frames': np.round(seq.astype(np.float64), 3).tolist()}


def history_view():
    h = sc._read_json(sc.HISTORY_PATH, [])
    return h if isinstance(h, list) else []


# ======================================================================= HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = 'SignTrainer/1.0'

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _host_ok(self):
        host = (self.headers.get('Host') or '').split(':')[0]
        return host in ('127.0.0.1', 'localhost')

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(min(n, 1_000_000)).decode('utf-8'))

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {'error': 'forbidden'})
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ('/', '/index.html'):
                with open(os.path.join(WEB_DIR, 'index.html'), 'rb') as f:
                    return self._send(200, f.read(), 'text/html; charset=utf-8')
            if u.path in ('/logo.png', '/favicon.png'):
                with open(os.path.join(WEB_DIR, u.path.lstrip('/')), 'rb') as f:
                    body = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'max-age=86400')
                self.end_headers()
                self.wfile.write(body)
                return None
            if u.path == '/frame.jpg':
                return self._frame(q.get('after', '-1'))
            if u.path == '/video.mjpg':
                return self._stream()
            if u.path == '/api/state':
                st = W.snapshot()
                st['train'] = {'running': TRAIN['running'], 'progress': TRAIN['progress']}
                return self._send(200, st)
            if u.path == '/api/words':
                return self._send(200, words_view())
            if u.path == '/api/vocab':
                alias = sc.vocab_alias()
                have = {alias.get(w['id']) for w in sc.load_words()}
                return self._send(200, [{**v, 'added': v['id'] in have} for v in sc.load_vocab()])
            if u.path == '/api/clips':
                return self._send(200, clips_view(q.get('word', '')))
            if u.path == '/api/clip':
                return self._send(200, clip_data(q.get('word', ''), q.get('file', '')))
            if u.path == '/api/train':
                return self._send(200, {k: TRAIN[k] for k in ('running', 'progress', 'log', 'result', 'error')})
            if u.path == '/api/history':
                return self._send(200, history_view())
            if u.path == '/api/settings':
                return self._send(200, {'values': sc.load_settings(), 'limits': sc.SETTING_LIMITS})
            if u.path == '/api/share':
                return self._send(200, sign_share.overview())
            if u.path == '/api/export':
                if q.get('consent') != '1':
                    raise ValueError('데이터 제공 동의에 체크해 주세요.')
                body, fname, info = sign_share.export_zip()
                print(f"exported {fname}: {info['clips']} clips, {info['words']} words, {info['people']} people", flush=True)
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{fname}"')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return None
            return self._send(404, {'error': 'not found'})
        except ValueError as e:
            return self._send(400, {'error': str(e)})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return None
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {'error': str(e)})

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, {'error': 'forbidden'})
        u = urlparse(self.path)
        try:
            if u.path == '/api/merge':          # body = one exported zip file
                n = int(self.headers.get('Content-Length') or 0)
                if n <= 0 or n > sign_share.MAX_ZIP_BYTES:
                    raise ValueError('파일이 비었거나 너무 커요 (100MB 초과).')
                data = self.rfile.read(n)
                name = parse_qs(u.query).get('name', [''])[0][:200]
                if TRAIN['running']:
                    raise ValueError('학습 중에는 합칠 수 없어요. 학습이 끝난 뒤 다시 해 주세요.')
                r = sign_share.merge_zip(data, name)
                W.refresh_words()
                print(f"merged {name}: +{r['added']} clips, {r['duplicate']} duplicate, {r['rejected']} rejected", flush=True)
                return self._send(200, r)
            b = self._json_body()
            if u.path == '/api/share/id':
                return self._send(200, {'me': sign_share.set_my_id(b.get('id'))})
            if u.path == '/api/reset':
                if b.get('confirm') != '초기화':
                    raise ValueError('확인 칸에 "초기화"라고 입력해 주세요.')
                if TRAIN['running']:
                    raise ValueError('학습 중에는 초기화할 수 없어요. 학습을 중지한 뒤 다시 해 주세요.')
                W.stop_record()
                with W.lock:
                    W.rec.update(recent=[], last=None)
                n = sc.reset_dataset()
                with _RATIO_LOCK:
                    _RATIO_CACHE.clear()
                sign_share.clear_cache()
                W.refresh_words()
                W.reload_model()
                print(f'dataset reset: {n} clip files deleted (temporary ID and settings kept)', flush=True)
                return self._send(200, {'deleted': n})
            if u.path == '/api/share/unblock':
                sign_share.unblock(b.get('contributor'))
                return self._send(200, {'ok': True})
            if u.path == '/api/share/forget':
                return self._send(200, {'cleared': sign_share.forget_deleted()})
            if u.path == '/api/share/remove':
                n = sign_share.remove_contributor(b.get('contributor'))
                W.refresh_words()
                return self._send(200, {'deleted': n})
            if u.path == '/api/words/add':
                if b.get('vocab'):
                    sc.add_vocab_word(b.get('vocab'))
                else:
                    sc.add_word(b.get('id'), b.get('name'), b.get('record_sec', 3.0))
                W.refresh_words()
                return self._send(200, words_view())
            if u.path == '/api/words/update':
                sc.update_word(b.get('id'), b.get('name'), b.get('record_sec'))
                W.refresh_words()
                return self._send(200, words_view())
            if u.path == '/api/words/delete':
                if W.rec['state'] != 'idle' and W.rec['word'] == b.get('id'):
                    W.stop_record()
                if b.get('id') != 'none' and sc.get_word(b.get('id')) is not None:
                    sign_share.remember_word(b.get('id'))
                sc.delete_word(b.get('id'))
                W.refresh_words()
                return self._send(200, words_view())
            if u.path == '/api/clips/delete':
                word = b.get('word')
                if sc.get_word(word) is None:
                    raise ValueError('없는 단어예요.')
                files = list(b.get('files') or [])
                n = sign_share.delete_clips(word, files)
                W.forget_clips(word, files)
                return self._send(200, {'deleted': n})
            if u.path == '/api/record/start':
                W.start_record(b.get('word'), b.get('auto', False))
                return self._send(200, {'ok': True})
            if u.path == '/api/record/stop':
                W.stop_record()
                return self._send(200, {'ok': True})
            if u.path == '/api/mode':
                W.set_mode(b.get('mode'))
                return self._send(200, {'ok': True})
            if u.path == '/api/train/start':
                start_training()
                return self._send(200, {'ok': True})
            if u.path == '/api/shutdown':
                threading.Thread(target=shutdown_app, daemon=True).start()
                return self._send(200, {'ok': True})
            if u.path == '/api/train/stop':
                TRAIN['stop'] = True
                return self._send(200, {'ok': True})
            if u.path == '/api/settings':
                s = sc.save_settings(b)
                W.apply_settings(s)
                return self._send(200, {'values': s, 'limits': sc.SETTING_LIMITS})
            return self._send(404, {'error': 'not found'})
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {'error': str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {'error': str(e)})

    def _frame(self, after):
        """one JPEG per request (short connection). waits up to 1 s for a frame newer than `after`.
        Unlike a never-ending MJPEG stream, this never uses up the browser's 6 connections per site."""
        try:
            after = int(after)
        except ValueError:
            after = -1
        with W.cond:
            W.cond.wait_for(lambda: W.frame_id != after and W.jpeg is not None, timeout=1.0)
            jpg, fid = W.jpeg, W.frame_id
        if jpg is None:
            return self._send(503, {'error': 'no frame yet'})
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(jpg)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Frame-Id', str(fid))
        self.end_headers()
        self.wfile.write(jpg)

    def _stream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        last = -1
        try:
            while True:
                with W.cond:
                    W.cond.wait_for(lambda: W.frame_id != last, timeout=2.0)
                    jpg, last = W.jpeg, W.frame_id
                if jpg is None:
                    continue
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                 + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return


def shutdown_app():
    """종료 button: stop training, release the webcam, then exit (1_web.bat closes its window on EXIT_FROM_WEB)

    [종료 규칙 - 바꾸지 말 것] exit code 99 = 1_web.bat closes the window. See README.md "종료 규칙".
    """
    time.sleep(0.4)                      # let the HTTP answer reach the browser first
    print('Shutting down (종료 button)...', flush=True)
    TRAIN['stop'] = True
    try:
        if W.reader is not None:
            W.reader.release()
    except Exception:  # noqa: BLE001
        pass
    os._exit(EXIT_FROM_WEB)


def main():
    print('speed: ' + sc.keep_full_speed(), flush=True)   # stay fast when the browser covers this window
    port = int(os.environ.get('SIGN_PORT', '8000'))
    srv = None
    for p in range(port, port + 10):
        try:
            srv = ThreadingHTTPServer((HOST, p), Handler)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        sys.exit('No free port between 8000 and 8009')
    srv.daemon_threads = True
    W.start()
    url = f'http://{HOST}:{port}/'
    print(f'Sign trainer running at {url}  (close this window to stop)')
    if os.environ.get('SIGN_NO_BROWSER') != '1':
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
