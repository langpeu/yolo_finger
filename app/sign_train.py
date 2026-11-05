# 수어 학습기 by Langpeu - (c) 2026 Langpeu, licensed under AGPL-3.0 (see LICENSE)
# Training: arm + finger shape, speed-varied copies, static and finger-swapped negatives
import time
import numpy as np
import torch
from sign_common import (load_words, load_settings, clip_files, load_clip, load_train_clip, fit_window, resample, hand_ratio,
                         add_shape_features, SignNet, HAND_BASE, FULL_DIM, MODEL_PATH, HISTORY_PATH,
                         _read_json, _write_json, clip_contributor, best_device, device_name)

MIN_CLIPS = 3        # a word needs at least this many clips to be trained
N_WARP = 4           # speed/timing-varied copies per real clip
N_SWAP = 1           # finger-swapped negatives per sign clip
SIMILAR = 0.35       # donor hand shapes closer than this to a sign's own shapes are not used
N_FREEZE = 2         # "same hand shape, no movement" negatives per MOVING sign clip
FREEZE_MIN_PATH = 0.8   # a sign counts as moving if the palm travels at least this far (shoulder widths)
POSE_ARM = (3, 4, 5, 6)  # elbows and wrists in the pose part of the features
VAL_SHARE = 0.2
BATCH = 64
HIDDEN = 96


class Stopped(Exception):
    pass


def _local_shape(p):  # (N,21,2) -> (N,42) wrist-relative, hand-size normalized
    size = np.maximum(np.linalg.norm(p[:, 9] - p[:, 0], axis=-1), 1e-3)
    return ((p - p[:, :1]) / size[:, None, None]).reshape(len(p), 42)


def _hand_points(clips, limit, rng):
    pts = []
    for c in clips:
        for s in range(2):
            m = c[:, HAND_BASE + 84 + s] > 0.5
            pts.extend(c[m, HAND_BASE + s * 42: HAND_BASE + s * 42 + 42].reshape(-1, 21, 2))
    if not pts:
        return np.zeros((0, 21, 2), np.float32)
    pts = np.stack(pts)
    if len(pts) > limit:
        pts = pts[rng.choice(len(pts), limit, replace=False)]
    return pts


def palm_path(c):
    """how far the palm travels in a clip (shoulder widths), max over both hands"""
    best = 0.0
    for s in range(2):
        m = c[:, HAND_BASE + 84 + s] > 0.5
        p = c[:, HAND_BASE + s * 42: HAND_BASE + s * 42 + 42].reshape(-1, 21, 2)[:, [0, 5, 9, 13, 17]].mean(1)
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ok = m[1:] & m[:-1]
        best = max(best, float(np.clip(d[ok], 0, 0.5).sum()))
    return best


def remove_motion(c, rng, mode='freeze'):
    """keep the hand SHAPE of every frame but take the movement away:
    'freeze' = hands stay at one spot, 'line' = hands slide in a straight line between two spots"""
    c = c.copy()
    pres = [c[:, HAND_BASE + 84 + s] > 0.5 for s in range(2)]
    any_p = np.flatnonzero(pres[0] | pres[1])
    if len(any_p) == 0:
        return None
    T = len(c)
    a, b = int(rng.choice(any_p)), int(rng.choice(any_p))
    w = np.zeros(T) if mode == 'freeze' else np.clip((np.arange(T) - T * 0.2) / (T * 0.6), 0, 1)
    for s in range(2):
        sl = slice(HAND_BASE + s * 42, HAND_BASE + s * 42 + 42)
        idx = np.flatnonzero(pres[s])
        if len(idx) == 0:
            continue
        ia = a if pres[s][a] else int(rng.choice(idx))
        ib = b if pres[s][b] else int(rng.choice(idx))
        p = c[:, sl].reshape(T, 21, 2).copy()
        target = (1 - w)[:, None] * p[ia, 0] + w[:, None] * p[ib, 0]          # where the wrist should be
        target = target + np.cumsum(rng.normal(0, 0.004, (T, 2)), 0)            # tiny natural sway
        p[pres[s]] = p[pres[s]] - p[pres[s]][:, :1] + target[pres[s]][:, None]
        c[:, sl] = p.reshape(T, 42)
    for i in POSE_ARM:
        pa, pb = c[a, 2 * i:2 * i + 2].copy(), c[b, 2 * i:2 * i + 2].copy()
        if np.any(pa != 0) and np.any(pb != 0):
            c[:, 2 * i:2 * i + 2] = (1 - w)[:, None] * pa + w[:, None] * pb
    return c.astype(np.float32)


def _warp(c, L, rng):
    """simulate doing the sign faster/slower or earlier/later in the window"""
    f = rng.uniform(0.7, 1.3)
    if f < 1.0:
        n = max(5, int(L * f))
        sq = resample(c, n)
        off = int(rng.integers(0, L - n + 1))
        out = np.concatenate([np.repeat(sq[:1], off, 0), sq, np.repeat(sq[-1:], L - n - off, 0)])
    else:
        n = max(5, int(L / f))
        st = int(rng.integers(0, L - n + 1))
        out = resample(c[st:st + n], L)
    return out.astype(np.float32)


def _augment(clips_by_label, labels, none_i, L, rng, log):
    """real clips -> (X list, y list) with all synthetic samples"""
    X, y = [], []
    for i, lab in enumerate(labels):
        for c in clips_by_label[lab]:
            X.append(c)
            y.append(i)
            for _ in range(N_WARP):
                X.append(_warp(c, L, rng))
                y.append(i)
    n_static = 0
    for lab in labels:
        for c in clips_by_label[lab]:
            t = rng.integers(len(c))
            s = np.repeat(c[t:t + 1], L, 0)
            s = s + rng.normal(0, 0.01, s.shape).astype(np.float32) * (s != 0)
            X.append(s.astype(np.float32))
            y.append(none_i)
            n_static += 1
    donors = _hand_points(clips_by_label['none'], 3000, rng)
    sign_pts = _hand_points([c for lab in labels if lab != 'none' for c in clips_by_label[lab]], 800, rng)
    if len(donors) and len(sign_pts):
        a, b = _local_shape(donors), _local_shape(sign_pts)
        d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T
        keep = np.sqrt(np.maximum(d2, 0) / 21).min(1) > SIMILAR
        donors = donors[keep]
    n_swap = 0
    if len(donors):
        for lab in labels:
            if lab == 'none':
                continue
            for c in clips_by_label[lab]:
                for _ in range(N_SWAP):
                    c2 = c.copy()
                    for s in range(2):
                        sl = slice(HAND_BASE + s * 42, HAND_BASE + s * 42 + 42)
                        m = c2[:, HAND_BASE + 84 + s] > 0.5
                        if not m.any():
                            continue
                        d = donors[rng.integers(len(donors))]
                        pts = c2[:, sl].reshape(-1, 21, 2).copy()
                        pts[m] = (d - d[0])[None] + pts[m][:, :1]
                        c2[:, sl] = pts.reshape(-1, 42)
                    X.append(c2)
                    y.append(none_i)
                    n_swap += 1
    # 3) "same hand shape, no movement" for signs that are defined by movement (e.g. drawing a circle)
    n_freeze = 0
    for lab in labels:
        if lab == 'none':
            continue
        for c in clips_by_label[lab]:
            if palm_path(c) < FREEZE_MIN_PATH:
                continue
            for k in range(N_FREEZE):
                c2 = remove_motion(c, rng, 'freeze' if k % 2 == 0 else 'line')
                if c2 is not None:
                    X.append(c2)
                    y.append(none_i)
                    n_freeze += 1
    log(f'  synthetic: +{N_WARP}x speed-varied, +{n_static} still, +{n_swap} finger-swapped ({len(donors)} donor hand shapes), '
        f'+{n_freeze} movement-removed')
    return X, y


def _fit(X, y, n_cls, epochs, device, log, progress, stop, p0, p1):
    Xt = torch.tensor(np.stack([add_shape_features(x) for x in X]), dtype=torch.float32).to(device)
    yt = torch.tensor(y).to(device)
    counts = torch.bincount(yt, minlength=n_cls).float().clamp(min=1)
    weights = counts.sum() / (n_cls * counts)
    model = SignNet(n_cls, HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(weight=weights)
    n = len(Xt)
    for ep in range(1, epochs + 1):
        if stop and stop():
            raise Stopped()
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, BATCH):          # mini-batch gradient descent (Adam)
            idx = perm[i:i + BATCH]
            xb = Xt[idx]
            mask = (xb != 0).float()
            scale = 1 + (torch.rand(len(xb), 1, 1, device=device) - 0.5) * 0.2
            xb = (xb * scale + 0.03 * torch.randn_like(xb)) * mask
            loss = lossf(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if progress:
            progress(p0 + (p1 - p0) * ep / epochs)
        if ep % 50 == 0 or ep == epochs:
            log(f'  epoch {ep}/{epochs}  loss {total / n:.4f}')
    model.eval()
    return model


def train_model(log=print, progress=None, stop=None):
    t_start = time.time()
    s = load_settings()
    L = int(round(s['window_sec'] * s['fps']))
    words = load_words()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    clips, counts, hands, skipped = {}, {}, {}, []
    people = {}                                   # word -> {contributor: clips}; '나' = my own recordings
    for w in words:
        files = clip_files(w['id'])
        if len(files) < MIN_CLIPS:
            if w['id'] != 'none':
                skipped.append(f"{w['name']}({len(files)})")
            continue
        lst = []
        for f in files:
            seq, sec = load_train_clip(w['id'], f)       # padded clips: only the part with the sign
            lst.append(fit_window(seq, sec, s['window_sec'], L))
            who = clip_contributor(f) or '나'
            people.setdefault(w['id'], {})
            people[w['id']][who] = people[w['id']].get(who, 0) + 1
        clips[w['id']] = lst
        counts[w['id']] = len(lst)
        hands[w['id']] = float(np.mean([hand_ratio(c) for c in lst]))
    names = {w['id']: w['name'] for w in words}
    log(f'window {s["window_sec"]:.1f}s / {L} frames, epochs {int(s["epochs"])}')
    for k in clips:
        log(f'  {names[k]} ({k}): {counts[k]} clips, hands visible {hands[k] * 100:.0f}%, people {len(people[k])}')
    everyone = sorted({p for k in clips for p in people[k]})
    log(f'  contributors: {len(everyone)} (' + ', '.join(everyone) + ')')
    if skipped:
        log(f'  skipped (fewer than {MIN_CLIPS} clips): ' + ', '.join(skipped))
    if 'none' not in clips:
        raise ValueError(f'"동작 없음" 클립이 {MIN_CLIPS}개 이상 필요해요.')
    labels = ['none'] + sorted(k for k in clips if k != 'none')
    if len(labels) < 2:
        raise ValueError(f'학습할 단어가 없어요. 단어 하나에 클립이 {MIN_CLIPS}개 이상 필요해요.')
    none_i = 0
    device = best_device()
    log('device: ' + device_name(device))

    # 1) validation run: hold out 20% of the REAL clips of each word (no augmented copies leak in)
    tr, va = {}, {}
    for lab in labels:
        idx = rng.permutation(len(clips[lab]))
        n_val = max(1, int(round(len(idx) * VAL_SHARE))) if len(idx) >= 5 else 0
        va[lab] = [clips[lab][i] for i in idx[:n_val]]
        tr[lab] = [clips[lab][i] for i in idx[n_val:]]
    log('[1/2] validation run (80% train / 20% held-out clips)')
    X, y = _augment(tr, labels, none_i, L, rng, log)
    model = _fit(X, y, len(labels), int(s['epochs']), device, log, progress, stop, 0.0, 0.45)
    conf = np.zeros((len(labels), len(labels)), int)
    vx = [c for lab in labels for c in va[lab]]
    vy = [i for i, lab in enumerate(labels) for _ in va[lab]]
    if vx:
        with torch.no_grad():
            xv = torch.tensor(np.stack([add_shape_features(c) for c in vx]), dtype=torch.float32).to(device)
            pred = model(xv).argmax(1).cpu().numpy()
        for t, p in zip(vy, pred):
            conf[t, p] += 1
    motion_check = {}
    for i, lab in enumerate(labels):
        moving = [c for c in va[lab] if lab != 'none' and palm_path(c) >= FREEZE_MIN_PATH]
        if not moving:
            continue
        fz = [remove_motion(c, rng, 'freeze') for c in moving]
        fz = [c for c in fz if c is not None]
        if fz:
            with torch.no_grad():
                xf = torch.tensor(np.stack([add_shape_features(c) for c in fz]), dtype=torch.float32).to(device)
                pf = model(xf).argmax(1).cpu().numpy()
            motion_check[lab] = float((pf == i).mean())
    per_word = {}
    for i, lab in enumerate(labels):
        n = conf[i].sum()
        per_word[lab] = None if n == 0 else float(conf[i, i] / n)
    val_acc = float(np.trace(conf) / conf.sum()) if conf.sum() else None
    log('  held-out accuracy: ' + ('n/a (too few clips)' if val_acc is None else f'{val_acc * 100:.0f}%') +
        '  [' + ', '.join(f'{names[l]} ' + ('-' if a is None else f'{a * 100:.0f}%') for l, a in per_word.items()) + ']')

    if motion_check:
        log('  motion check (held-out moving signs with the movement removed, wrongly recognized): ' +
            ', '.join(f'{names[l]} {v * 100:.0f}%' for l, v in motion_check.items()) + '  (lower is better)')

    # 2) final model on ALL clips
    log('[2/2] final model on all clips')
    X, y = _augment(clips, labels, none_i, L, rng, log)
    model = _fit(X, y, len(labels), int(s['epochs']), device, log, progress, stop, 0.45, 1.0)
    torch.save({'state': model.state_dict(), 'labels': labels, 'dim': FULL_DIM, 'seq_len': L,
                'window_sec': float(s['window_sec']), 'hidden': HIDDEN,
                'contributors': {l: people[l] for l in labels}}, MODEL_PATH)
    result = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'labels': labels,
        'names': {l: names[l] for l in labels},
        'counts': {l: counts[l] for l in labels},
        'contributors': {l: people[l] for l in labels},
        'people': len(everyone),
        'hands': {l: hands[l] for l in labels},
        'val_acc': val_acc,
        'per_word': per_word,
        'confusion': conf.tolist(),
        'motion_check': motion_check,
        'epochs': int(s['epochs']),
        'window_sec': float(s['window_sec']),
        'samples': len(X),
        'duration_s': round(time.time() - t_start, 1),
    }
    hist = _read_json(HISTORY_PATH, [])
    if not isinstance(hist, list):
        hist = []
    hist.append(result)
    _write_json(HISTORY_PATH, hist[-100:])
    log(f'saved model -> sign_model.pt  ({result["duration_s"]} s)')
    return result


if __name__ == '__main__':
    train_model()
