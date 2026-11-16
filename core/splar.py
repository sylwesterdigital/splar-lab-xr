#!/usr/bin/env python3
"""SPLAR — single-file Flask/WebGL room reconstruction POC.

Python: Flask + SQLite + Pillow + NumPy, optional VGGT (real multi-view 3D).
Frontend: embedded HTML/CSS/JS; custom WebGL2 Gaussian splat renderer, no JS CDN.
"""
from __future__ import annotations

import io
import json
import math
import os

# The first 5 GB model download can fail in hf_xet on some macOS systems.
# Apply before any optional huggingface_hub / VGGT imports.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import secrets
import shutil
import sqlite3
import struct
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
STORE = ROOT / 'splar_data'
SCENES = STORE / 'scenes'
DATABASE = STORE / 'splar.sqlite3'
STORE.mkdir(exist_ok=True)
SCENES.mkdir(exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='splar')
model_lock = threading.Lock()
_model = None
_model_device = None

SPLAT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('radius', '<f4'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
])
MAGIC = b'SPLAR01\x00'
MAX_SPLATS = max(1000, int(os.environ.get('SPLAR_MAX_SPLATS', '120000')))
# The photo gallery has no hard maximum. Reconstruction processes overlapping
# windows so input photos do not all have to reside on the GPU simultaneously.
BATCH_VIEWS = max(4, int(os.environ.get('SPLAR_BATCH_VIEWS', '10')))
BATCH_OVERLAP = min(BATCH_VIEWS - 2, max(2, int(os.environ.get('SPLAR_BATCH_OVERLAP', '2'))))


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def db():
    conn = sqlite3.connect(DATABASE, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=30000')
    return conn


def init_db():
    with db() as c:
        c.execute('PRAGMA journal_mode=WAL')
        c.executescript('''
        CREATE TABLE IF NOT EXISTS scenes (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL,
            stage TEXT NOT NULL, error TEXT, point_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY, scene_id TEXT NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
            filename TEXT NOT NULL, path TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_images_scene ON images(scene_id);
        ''')
        c.execute("UPDATE scenes SET status='error', stage='Interrupted', error='Server restarted during reconstruction. Process again.' WHERE status='processing'")


def scene_path(scene_id):
    return SCENES / scene_id


def fetch_scene(scene_id):
    with db() as c:
        s = c.execute('SELECT * FROM scenes WHERE id=?', (scene_id,)).fetchone()
        if s is None:
            abort(404, 'Scene not found')
        ims = c.execute('SELECT id, filename, created_at FROM images WHERE scene_id=? ORDER BY created_at, rowid', (scene_id,)).fetchall()
    result = dict(s)
    result['images'] = [dict(i) | {'url': f'/api/images/{i["id"]}'} for i in ims]
    result['splat_url'] = f'/api/scenes/{scene_id}/splat' if s['status'] == 'ready' else None
    result['ply_url'] = f'/api/scenes/{scene_id}/ply' if s['status'] == 'ready' else None
    return result


def set_scene(scene_id, **kwargs):
    kwargs['updated_at'] = now()
    with db() as c:
        c.execute('UPDATE scenes SET ' + ', '.join(f'{k}=?' for k in kwargs) + ' WHERE id=?',
                  tuple(kwargs.values()) + (scene_id,))


def json_error(message, status=400):
    return jsonify({'error': str(message)}), status


def write_splats(path: Path, vertices, colors, radii, alpha=235):
    n = len(vertices)
    out = np.empty(n, dtype=SPLAT_DTYPE)
    for j, name in enumerate(('x', 'y', 'z')):
        out[name] = vertices[:, j].astype(np.float32)
    out['radius'] = np.asarray(radii, dtype=np.float32)
    for j, name in enumerate(('r', 'g', 'b')):
        out[name] = colors[:, j].astype(np.uint8)
    out['a'] = alpha
    with path.open('wb') as f:
        f.write(struct.pack('<8sI', MAGIC, n))
        f.write(out.tobytes())
    return n


def make_demo():
    """A deterministic complete illustrative 3D room. Not a reconstruction."""
    output = STORE / 'demo.splat'
    if output.exists():
        return
    rng = np.random.default_rng(39)
    pieces = []
    # Each plane is a physically placed 3D rectangular surface, not a flat image.
    def plane(origin, axis_u, axis_v, nu, nv, color, noise=5, radius=.044):
        uu, vv = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
        p = np.asarray(origin)[None, None, :] + uu[..., None] * np.asarray(axis_u) + vv[..., None] * np.asarray(axis_v)
        p += rng.normal(0, .006, p.shape)
        n = nu * nv
        rgb = np.broadcast_to(np.asarray(color, dtype=float), (n, 3)).copy()
        light = (p[..., 0].ravel() * 1.5 + p[..., 1].ravel() * 1.5)
        rgb += light[:, None] + rng.normal(0, noise, (n, 3))
        pieces.append((p.reshape(-1, 3).astype(np.float32), np.clip(rgb, 0, 255).astype(np.uint8), np.full(n, radius, dtype=np.float32)))
    # Shell / floor
    plane((-3.8, -1.28, -3.0), (7.6, 0, 0), (0, 0, 6.5), 112, 93, (101, 89, 76), 7, .054)
    plane((-3.8, -1.28, -3.0), (7.6, 0, 0), (0, 4.3, 0), 120, 65, (180, 171, 156), 6, .047)
    plane((-3.8, -1.28, 3.5), (0, 0, -6.5), (0, 4.3, 0), 95, 65, (147, 143, 132), 6, .05)
    # Rug
    plane((-2.0, -1.255, -.4), (3.8, 0, 0), (0, 0, 2.85), 80, 58, (159, 121, 91), 9, .054)
    plane((-1.81, -1.251, -.23), (3.4, 0, 0), (0, 0, 2.55), 65, 50, (177, 138, 102), 7, .055)
    # Couch seat + back and arms
    plane((-2.8, -.53, -2.45), (3.1, 0, 0), (0, 0, 1.28), 70, 27, (97, 116, 117), 7)
    plane((-2.8, -.53, -2.48), (3.1, 0, 0), (0, 1.12, 0), 70, 28, (79, 105, 106), 7)
    plane((-2.8, -.53, -2.43), (0, 0, 1.32), (0, .57, 0), 27, 13, (79, 100, 100))
    plane((.25, -.53, -2.43), (0, 0, 1.32), (0, .57, 0), 27, 13, (79, 100, 100))
    # Coffee table
    plane((-.9, -.49, .25), (2.05, 0, 0), (0, 0, .8), 48, 18, (82, 66, 56), 5)
    for x in (-.78, 1.02):
        for z in (.32, .94):
            plane((x, -1.23, z), (0, .74, 0), (.075, 0, 0), 16, 4, (61, 54, 49))
    # Floating art and large window
    plane((1.05, .07, -2.975), (1.85, 0, 0), (0, 1.72, 0), 36, 40, (72, 94, 105), 5)
    plane((1.16, .18, -2.959), (1.63, 0, 0), (0, 1.5, 0), 34, 35, (111, 140, 151), 9)
    plane((1.87, .08, -2.94), (.08, 0, 0), (0, 1.7, 0), 4, 38, (51, 55, 60), 1)
    plane((1.1, .88, -2.93), (1.7, 0, 0), (0, .06, 0), 34, 3, (50, 55, 61), 1)
    plane((-2.64, .76, -2.962), (1.4, 0, 0), (0, 1.18, 0), 26, 26, (143, 112, 84), 6)
    plane((-2.51, .88, -2.949), (1.14, 0, 0), (0, .92, 0), 24, 21, (190, 152, 113), 9)
    # Tall sideboard and decorative vertical fixture
    plane((2.5, -1.23, -1.9), (.89, 0, 0), (0, 1.15, 0), 24, 26, (90, 75, 63), 6)
    plane((2.5, -.1, -1.9), (.89, 0, 0), (0, 0, 1.2), 24, 25, (102, 84, 70), 6)
    plane((3.1, -.1, .7), (0, 0, .08), (0, 1.55, 0), 5, 38, (180, 156, 100), 4)
    # Light shade
    for j in range(18):
        ang = j / 18 * math.tau
        x, z = 3.12 + .35 * math.cos(ang), .7 + .35 * math.sin(ang)
        plane((x, 1.28, z), (.055, 0, 0), (0, .43, 0), 2, 11, (217, 181, 127), 5, .04)
    p = np.concatenate([x[0] for x in pieces])
    c = np.concatenate([x[1] for x in pieces])
    r = np.concatenate([x[2] for x in pieces])
    write_splats(output, p, c, r)


def load_vggt():
    global _model, _model_device
    with model_lock:
        if _model is not None:
            return _model, _model_device
        import torch
        from vggt.models.vggt import VGGT
        choice = os.environ.get('SPLAR_DEVICE', '').lower().strip()
        if choice in ('cpu', 'mps', 'cuda'):
            device = choice
        elif torch.cuda.is_available():
            device = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f'[SPLAR] Loading VGGT-1B on {device} (first run downloads model weights)...', flush=True)
        model = VGGT.from_pretrained('facebook/VGGT-1B')
        model.eval().to(device)
        _model, _model_device = model, device
        return model, device


def fit_similarity(source, target):
    """Robustly map 3D points from a new VGGT window to the prior window.

    Both arrays contain matched pixels from the same overlapping input photos.
    A similarity transform (rotation + translation + scale) corrects each
    inference window's arbitrary coordinate system.
    """
    if len(source) < 100:
        raise RuntimeError('Cannot align photo batches: insufficient overlapping geometry.')
    rng = np.random.default_rng(71)
    if len(source) > 12000:
        picks = rng.choice(len(source), 12000, replace=False)
        source, target = source[picks], target[picks]

    def umeyama(src, dst):
        a, b = src.mean(axis=0), dst.mean(axis=0)
        x, y = src - a, dst - b
        covariance = y.T @ x / len(x)
        u, singular, vt = np.linalg.svd(covariance)
        sign = np.eye(3)
        sign[-1, -1] = np.sign(np.linalg.det(u @ vt))
        rotation = u @ sign @ vt
        variance = float(np.sum(x * x) / len(x))
        if variance < 1e-10:
            raise RuntimeError('Cannot align photo batches: overlapping geometry is degenerate.')
        scale = float(np.sum(singular * np.diag(sign)) / variance)
        translation = b - scale * (rotation @ a)
        return scale, rotation, translation

    # Robustly re-fit while discarding discordant point-map pixels.
    src, dst = source.astype(np.float64), target.astype(np.float64)
    for _ in range(3):
        scale, rotation, translation = umeyama(src, dst)
        residual = np.linalg.norm((source @ rotation.T) * scale + translation - target, axis=1)
        cutoff = max(float(np.quantile(residual, 0.70)), 1e-5)
        keep = residual <= cutoff
        if np.count_nonzero(keep) < 100:
            break
        src, dst = source[keep], target[keep]
    scale, rotation, translation = umeyama(src, dst)
    if not np.isfinite(scale) or scale <= 0 or scale > 1000 or scale < .001:
        raise RuntimeError('Cannot align photo batches: inconsistent reconstruction scale.')
    return scale, rotation, translation


def align_window(points, confidence, indices, anchors):
    """Align a new multi-view inference window using shared, same-pixel points."""
    sources, targets = [], []
    for local, global_id in enumerate(indices):
        if global_id not in anchors:
            continue
        previous, previous_conf = anchors[global_id]
        current, current_conf = points[local], confidence[local]
        if previous.shape != current.shape:
            raise RuntimeError('VGGT returned differing resolutions for overlapping photos.')
        stride = max(1, min(current.shape[:2]) // 42)
        a, b = current[::stride, ::stride], previous[::stride, ::stride]
        ca, cb = current_conf[::stride, ::stride], previous_conf[::stride, ::stride]
        valid = (np.isfinite(a).all(axis=2) & np.isfinite(b).all(axis=2)
                 & np.isfinite(ca) & np.isfinite(cb))
        valid &= (ca >= np.quantile(ca[np.isfinite(ca)], .30)) if np.any(np.isfinite(ca)) else False
        valid &= (cb >= np.quantile(cb[np.isfinite(cb)], .30)) if np.any(np.isfinite(cb)) else False
        valid &= (np.linalg.norm(a, axis=2) > 1e-4) & (np.linalg.norm(b, axis=2) > 1e-4)
        sources.append(a[valid])
        targets.append(b[valid])
    if not sources or sum(len(x) for x in sources) < 100:
        raise RuntimeError('Photo batches could not be aligned. Use overlapping consecutive photos.')
    scale, rotation, translation = fit_similarity(np.concatenate(sources), np.concatenate(targets))
    points = (points @ rotation.T) * scale + translation
    return points.astype(np.float32), scale


def window_samples(points, confidence, rgb, start_view, world_scale, budget):
    """Filter and sample newly reconstructed image views from one window."""
    points = points[start_view:]
    confidence = confidence[start_view:]
    rgb = rgb[start_view:]
    if len(points) == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=np.float32))
    n, h, w, _ = points.shape
    stride = max(1, int(math.ceil(math.sqrt(n * h * w / max(1000, budget)))))
    pts = points[:, ::stride, ::stride, :]
    cf = confidence[:, ::stride, ::stride]
    colors = rgb[:, ::stride, ::stride, :]
    dx = np.linalg.norm(np.diff(pts, axis=2, append=pts[:, :, -1:, :]), axis=3)
    dy = np.linalg.norm(np.diff(pts, axis=1, append=pts[:, -1:, :, :]), axis=3)
    footprint = .5 * (dx + dy)
    p = pts.reshape(-1, 3)
    color = colors.reshape(-1, 3)
    conf = cf.reshape(-1)
    radius = footprint.reshape(-1)
    valid = np.isfinite(p).all(axis=1) & np.isfinite(conf) & np.isfinite(radius)
    valid &= np.linalg.norm(p, axis=1) > 1e-5
    if valid.sum() < 100:
        return p[:0], color[:0], radius[:0]
    threshold = np.quantile(conf[valid], .25)
    valid &= conf >= threshold
    lo, hi = np.quantile(p[valid], [.01, .99], axis=0)
    valid &= np.all((p >= lo) & (p <= hi), axis=1)
    return p[valid], color[valid], radius[valid] * abs(world_scale)


def reconstruct(scene_id):
    try:
        with db() as c:
            rows = c.execute('SELECT path FROM images WHERE scene_id=? ORDER BY created_at, rowid', (scene_id,)).fetchall()
        total = len(rows)
        if total < 2:
            raise RuntimeError('Upload at least two photos of the same room.')
        set_scene(scene_id, stage='Loading geometry model')
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        model, device = load_vggt()

        anchors = {}  # last overlapping views, retained in the aligned world frame
        rng = np.random.default_rng(7)
        kept_points = np.empty((0, 3), dtype=np.float32)
        kept_colors = np.empty((0, 3), dtype=np.uint8)
        kept_radii = np.empty((0,), dtype=np.float32)
        kept_priority = np.empty((0,), dtype=np.float64)
        start = 0
        while start < total:
            end = min(total, start + BATCH_VIEWS)
            ids = list(range(start, end))
            start_view = sum(i in anchors for i in ids)
            set_scene(scene_id, stage=f'Reconstructing photos {start + 1}–{end} of {total}')
            paths = [str(scene_path(scene_id) / rows[i]['path']) for i in ids]
            images = load_and_preprocess_images(paths).to(device)
            with torch.inference_mode():
                if device == 'cuda':
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model(images)
                else:
                    result = model(images)
            points = result['world_points'][0].detach().float().cpu().numpy()
            confidence = result['world_points_conf'][0].detach().float().cpu().numpy()
            rgb = (images.detach().float().cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
            del result, images
            if anchors:
                set_scene(scene_id, stage=f'Aligning photos {start + 1}–{end} of {total}')
                points, world_scale = align_window(points, confidence, ids, anchors)
            else:
                world_scale = 1.0
            # Next window shares these two photos; correspondence fixes scale + pose.
            anchors = {ids[k]: (points[k].copy(), confidence[k].copy())
                       for k in range(max(0, len(ids) - BATCH_OVERLAP), len(ids))}
            p, color, radius = window_samples(points, confidence, rgb, start_view, world_scale,
                                              min(MAX_SPLATS, 120000))
            if len(p):
                # Priority reservoir maintains a global splat budget with no
                # limit on the number of images / inference windows.
                priorities = rng.random(len(p))
                p = np.concatenate((kept_points, p))
                color = np.concatenate((kept_colors, color))
                radius = np.concatenate((kept_radii, radius))
                priorities = np.concatenate((kept_priority, priorities))
                if len(p) > MAX_SPLATS:
                    choose = np.argpartition(priorities, -MAX_SPLATS)[-MAX_SPLATS:]
                    p, color, radius, priorities = (p[choose], color[choose],
                                                     radius[choose], priorities[choose])
                kept_points, kept_colors, kept_radii, kept_priority = p, color, radius, priorities
            if device == 'mps' and hasattr(torch, 'mps'):
                torch.mps.empty_cache()
            if end >= total:
                break
            start = end - BATCH_OVERLAP
        p, color, radius = kept_points, kept_colors, kept_radii
        if len(p) < 500:
            raise RuntimeError('Insufficient valid 3D points. Use overlapping, well-lit photos.')
        set_scene(scene_id, stage='Building Gaussian splats')
        center = .5 * (np.quantile(p, .05, axis=0) + np.quantile(p, .95, axis=0))
        size = np.quantile(p, .95, axis=0) - np.quantile(p, .05, axis=0)
        scale = 6.6 / max(float(np.max(size)), .01)
        p = (p - center) * scale
        p[:, 1] *= -1
        p[:, 2] *= -1
        radii = np.clip(radius * scale * .62, .018, .15)
        count = write_splats(scene_path(scene_id) / 'scene.splat', p, color, radii)
        set_scene(scene_id, status='ready', stage='Ready', error=None, point_count=count)
        print(f'[SPLAR] scene={scene_id} ready, {total} photos, {count:,} Gaussian splats', flush=True)
    except Exception as exc:
        traceback.print_exc()
        error = f'{type(exc).__name__}: {exc}'
        if 'Background writer channel closed' in error:
            error += ' — Model download/cache write failed. Check free disk space and restart with HF_HUB_DISABLE_XET=1 ./run.zsh.'
        set_scene(scene_id, status='error', stage='Failed', error=error[:700])


@app.get('/')
def home():
    return Response(INDEX_HTML, mimetype='text/html')


@app.get('/api/health')
def health():
    return jsonify({'ok': True, 'engine': 'VGGT-1B / Gaussian splat renderer', 'database': 'SQLite'})


@app.get('/api/version')
def app_version():
    """Current deployed code version; never cached by the browser."""
    version_file = ROOT.parent / 'VERSION'
    version = version_file.read_text().strip() if version_file.is_file() else '0.1.0'
    response = jsonify({'version': version})
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response


@app.get('/api/demo/splat')
def demo_splat():
    return send_file(STORE / 'demo.splat', mimetype='application/octet-stream', max_age=3600)


@app.get('/api/scenes')
def scenes_list():
    with db() as c:
        rows = c.execute('SELECT id FROM scenes ORDER BY updated_at DESC').fetchall()
    return jsonify([fetch_scene(r['id']) for r in rows])


@app.post('/api/scenes')
def scenes_new():
    with db() as c:
        count = c.execute('SELECT COUNT(*) AS n FROM scenes').fetchone()['n']
        sid = secrets.token_hex(9)
        title = (request.get_json(silent=True) or {}).get('title', '')
        title = str(title).strip()[:70] or f'Room {count + 1:02d}'
        c.execute('INSERT INTO scenes(id,title,status,stage,error,created_at,updated_at) VALUES (?,?,?,?,?,?,?)',
                  (sid, title, 'draft', 'Awaiting photos', None, now(), now()))
    scene_path(sid).mkdir(parents=True, exist_ok=True)
    return jsonify(fetch_scene(sid)), 201


@app.get('/api/scenes/<scene_id>')
def scenes_one(scene_id):
    return jsonify(fetch_scene(scene_id))


@app.post('/api/scenes/<scene_id>/images')
def add_images(scene_id):
    s = fetch_scene(scene_id)
    if s['status'] == 'processing':
        return json_error('Cannot add photos while processing.', 409)
    files = request.files.getlist('files')
    if not files:
        return json_error('No photos provided.')
    output = scene_path(scene_id)
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        for file in files:
            raw = file.read()
            if not raw or len(raw) > 16 * 1024 * 1024:
                raise ValueError('Each image must be between 1 byte and 16 MB.')
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.format not in ('JPEG', 'PNG', 'WEBP'):
                    raise ValueError('Upload JPG, PNG, or WebP photos.')
                img = ImageOps.exif_transpose(opened).convert('RGB')
                if min(img.size) < 256:
                    raise ValueError('Photos must be at least 256 px wide and tall.')
                img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                iid = secrets.token_hex(9)
                fname = f'{iid}.jpg'
                img.save(output / fname, 'JPEG', quality=90, optimize=True)
                saved.append((iid, scene_id, Path(file.filename or 'photo').name[:100], fname, now()))
    except (ValueError, UnidentifiedImageError, OSError) as exc:
        for row in saved:
            (output / row[3]).unlink(missing_ok=True)
        return json_error(str(exc))
    with db() as c:
        c.executemany('INSERT INTO images (id,scene_id,filename,path,created_at) VALUES (?,?,?,?,?)', saved)
        c.execute("UPDATE scenes SET status='draft', stage='Ready to process', error=NULL, point_count=0, updated_at=? WHERE id=?", (now(), scene_id))
    (output / 'scene.splat').unlink(missing_ok=True)
    return jsonify(fetch_scene(scene_id)), 201


@app.get('/api/images/<image_id>')
def image_file(image_id):
    with db() as c:
        row = c.execute('SELECT * FROM images WHERE id=?', (image_id,)).fetchone()
    if row is None:
        abort(404)
    return send_file(scene_path(row['scene_id']) / row['path'], mimetype='image/jpeg', max_age=3600)


@app.delete('/api/scenes/<scene_id>/images/<image_id>')
def delete_image(scene_id, image_id):
    s = fetch_scene(scene_id)
    if s['status'] == 'processing':
        return json_error('Cannot remove photos while processing.', 409)
    with db() as c:
        row = c.execute('SELECT path FROM images WHERE id=? AND scene_id=?', (image_id, scene_id)).fetchone()
        if row is None:
            abort(404)
        c.execute('DELETE FROM images WHERE id=? AND scene_id=?', (image_id, scene_id))
        c.execute("UPDATE scenes SET status='draft', stage='Ready to process', point_count=0, error=NULL, updated_at=? WHERE id=?", (now(), scene_id))
    (scene_path(scene_id) / row['path']).unlink(missing_ok=True)
    (scene_path(scene_id) / 'scene.splat').unlink(missing_ok=True)
    return jsonify(fetch_scene(scene_id))


@app.post('/api/scenes/<scene_id>/process')
def process(scene_id):
    s = fetch_scene(scene_id)
    if s['status'] == 'processing':
        return json_error('Already processing.', 409)
    if len(s['images']) < 2:
        return json_error('Add at least 2 photos of the same room.')
    set_scene(scene_id, status='processing', stage='Queued', error=None, point_count=0)
    pool.submit(reconstruct, scene_id)
    return jsonify(fetch_scene(scene_id)), 202


@app.delete('/api/scenes/<scene_id>')
def delete_scene(scene_id):
    s = fetch_scene(scene_id)
    if s['status'] == 'processing':
        return json_error('Cannot delete while processing.', 409)
    with db() as c:
        c.execute('DELETE FROM scenes WHERE id=?', (scene_id,))
    shutil.rmtree(scene_path(scene_id), ignore_errors=True)
    return jsonify({'ok': True})


@app.get('/api/scenes/<scene_id>/splat')
def scene_splat(scene_id):
    s = fetch_scene(scene_id)
    if s['status'] != 'ready':
        abort(404)
    return send_file(scene_path(scene_id) / 'scene.splat', mimetype='application/octet-stream', max_age=60)


@app.get('/api/scenes/<scene_id>/ply')
def scene_ply(scene_id):
    s = fetch_scene(scene_id)
    if s['status'] != 'ready':
        abort(404)
    path = scene_path(scene_id) / 'scene.splat'
    raw = path.read_bytes()
    magic, n = struct.unpack_from('<8sI', raw, 0)
    if magic != MAGIC or len(raw) != 12 + n * SPLAT_DTYPE.itemsize:
        abort(500)
    # Binary PLY, columns match the internal 20-byte vertex layout.
    header = (f'ply\nformat binary_little_endian 1.0\n'
              f'comment SPLAR isotropic Gaussian splats; radius is in scene units\n'
              f'element vertex {n}\n'
              f'property float x\nproperty float y\nproperty float z\n'
              f'property float radius\nproperty uchar red\nproperty uchar green\n'
              f'property uchar blue\nproperty uchar alpha\nend_header\n').encode()
    return Response(header + raw[12:], mimetype='application/octet-stream', headers={
        'Content-Disposition': f'attachment; filename="SPLAR-{scene_id}.ply"'})


@app.errorhandler(413)
def large(_):
    return json_error('Upload too large (80 MB per request).', 413)


# Embedded frontend (HTML, CSS and JS) follows. The entire app is this one Python file.
INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#101214"><title>SPLAR — Spatial Capture</title>
<style>
:root{color-scheme:dark;--bg:#0c0e10;--panel:#181b1deF;--surface:#252a2b;--edge:rgba(255,255,255,.105);--text:#ecefea;--muted:#929a99;--accent:#d6f477;--quiet:#c4c9c2;--radius:16px}
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg);color:var(--text);font:13px/1.5 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Inter,sans-serif;-webkit-font-smoothing:antialiased}button,input{font:inherit}button{cursor:pointer}button:disabled{cursor:not-allowed;opacity:.38}button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:3px}::selection{background:#d7f37c;color:#101510}
#viewport{position:absolute;inset:0;background:radial-gradient(ellipse at 49% 53%,#252b2c 0,#171b1d 41%,#0e1012 76%);overflow:hidden}#viewport::after{content:'';position:absolute;inset:0;pointer-events:none;opacity:.24;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);background-size:64px 64px;mask-image:linear-gradient(to bottom,transparent,#000 30%,transparent 90%)}#glCanvas{width:100%;height:100%;display:block;touch-action:none;cursor:grab;position:relative;z-index:1}#glCanvas.dragging{cursor:grabbing}
.topbar{position:absolute;z-index:20;top:0;left:0;right:0;height:78px;padding:0 32px;display:flex;align-items:center;justify-content:space-between;pointer-events:none;background:linear-gradient(#0d0f11f5,#0d0f11b0 69%,transparent)}.topbar>*{pointer-events:auto}.brand{display:flex;align-items:center;gap:11px;color:var(--text);user-select:none}.logo{height:33px;width:33px}.brandname{font-size:20px;letter-spacing:.135em;font-weight:760;line-height:1}.brandcaption{border-left:1px solid #484d4c;padding-left:14px;margin-left:8px;color:var(--muted);font-size:10px;letter-spacing:.14em;text-transform:uppercase;font-weight:700}.topright{display:flex;gap:9px;align-items:center}.statuspill{display:flex;gap:8px;align-items:center;color:#aeb7b1;font-size:11px;letter-spacing:.06em;margin-right:17px}.statusdot{width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 14px #d6f47770}.topbtn,.iconbtn{display:inline-flex;gap:8px;align-items:center;justify-content:center;border:1px solid var(--edge);border-radius:9px;color:var(--text);background:rgba(40,44,45,.66);height:35px;padding:0 13px;font-size:11px;font-weight:650;letter-spacing:.01em}.topbtn:hover,.iconbtn:hover{background:#333938}.topbtn.green{background:var(--accent);color:#161d0e;border-color:var(--accent);font-weight:750}.topbtn.green:hover,.primary:hover{background:#e6ffa2}.ico{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex:none}
.panel{position:absolute;z-index:12;width:315px;top:102px;left:27px;background:var(--panel);border:1px solid #393e3d;border-radius:var(--radius);box-shadow:0 30px 70px #0009,0 4px 20px #0004;backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);overflow:hidden;max-height:calc(100vh - 140px);display:flex;flex-direction:column}.panel.right{left:auto;right:27px;width:250px;top:122px}.panel-head{height:49px;min-height:49px;display:flex;align-items:center;justify-content:space-between;padding:0 15px 0 18px;border-bottom:1px solid var(--edge);cursor:move;touch-action:none;user-select:none}.panel-title{font-size:11px;letter-spacing:.14em;font-weight:770;text-transform:uppercase;display:flex;align-items:center;gap:9px}.panel-title .head-mark{height:7px;width:7px;border-radius:2px;background:var(--accent);transform:rotate(45deg)}.panel-toggle{border:0;background:transparent;color:#abb5b0;padding:7px;border-radius:6px;display:flex;align-items:center}.panel-toggle:hover{color:white;background:#ffffff15}.panel-body{overflow:auto;padding:19px 17px 17px;scrollbar-width:thin;scrollbar-color:#45504a transparent}.panel.collapsed .panel-body{display:none}.panel.collapsed .panel-head{border:0}
.section-label{display:flex;align-items:center;justify-content:space-between;margin-bottom:11px;font-weight:720;color:#dfe4dd;font-size:11px}.micro{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);font-weight:700}.counter{font-variant-numeric:tabular-nums;color:var(--muted);font-size:10px}.drop{border:1px dashed #555f57;border-radius:12px;background:#202522b8;text-align:center;padding:21px 11px 17px;cursor:pointer;transition:.16s}.drop:hover,.drop.dragover{border-color:var(--accent);background:#293324}.drop-graphic{width:40px;height:40px;border:1px solid #697265;border-radius:12px;display:grid;place-items:center;margin:0 auto 10px;color:var(--accent)}.drop strong{font-size:12px;font-weight:720;display:block}.drop small{display:block;font-size:10px;color:var(--muted);margin-top:4px}.thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:13px 0 16px}.thumb{aspect-ratio:1.13/1;border-radius:9px;position:relative;background:#292d2d;overflow:hidden;border:1px solid #434b46}.thumb img{width:100%;height:100%;object-fit:cover}.thumb .remove{position:absolute;right:4px;top:4px;width:19px;height:19px;border-radius:5px;border:0;background:#151919df;color:#f5f5f5;display:grid;place-items:center;font-size:15px;line-height:1}.thumb.add{border:1px dashed #707a70;background:#222723;display:grid;place-items:center;color:var(--accent);cursor:pointer}.thumb.add:hover{border-color:var(--accent);background:#303a2a}.thumb.add span{font-size:27px;font-weight:250;line-height:1}.inline-note{color:var(--muted);font-size:10px}.primary{display:flex;gap:9px;align-items:center;justify-content:center;width:100%;border:0;border-radius:10px;background:var(--accent);color:#161d10;font-weight:770;font-size:12px;letter-spacing:.02em;height:43px}.primary svg{width:16px;height:16px}.process-meta{margin-top:11px;display:flex;justify-content:space-between;color:var(--muted);font-size:10px}.meter{margin-top:9px;height:3px;border-radius:4px;background:#353d39;overflow:hidden}.meter>div{height:100%;width:0;background:var(--accent);transition:width .6s}.meter.loading>div{width:30%;animation:loading 1.5s ease-in-out infinite}@keyframes loading{0%{transform:translateX(-120%)}100%{transform:translateX(410%)}}
.rowline{display:flex;align-items:center;justify-content:space-between;padding:9px 0;border-bottom:1px solid #ffffff12;gap:9px}.rowline:last-child{border:0}.rowline .label{font-size:11px;color:#939d97}.rowline .value{font-size:11px;font-weight:690;font-variant-numeric:tabular-nums;text-align:right}.section-divider{height:1px;background:var(--edge);margin:18px 0}.range{accent-color:var(--accent);width:100%;display:block;margin:13px 0 5px;cursor:pointer}.detail{font-size:10px;color:var(--muted);line-height:1.65}.link-btn{width:100%;height:36px;background:#292e2c;border:1px solid #404840;color:var(--text);border-radius:8px;display:flex;align-items:center;justify-content:center;gap:8px;font-size:11px;font-weight:650;margin-top:12px}.link-btn:hover{background:#353d38}.link-btn.hidden{display:none}
.scene-heading{position:absolute;top:105px;left:50%;transform:translateX(-50%);pointer-events:none;text-align:center;z-index:2;white-space:nowrap}.scene-heading .eyebrow{font-size:10px;letter-spacing:.2em;color:#b7c1b6;font-weight:700}.scene-heading h1{margin:5px 0;font-size:26px;font-weight:550;letter-spacing:-.065em}.scene-heading p{font-size:11px;color:var(--muted);margin:0}.scene-heading.condensed{opacity:.38}.scene-heading.condensed h1{font-size:20px}
.bottom{position:absolute;z-index:9;bottom:25px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:4px;padding:5px;border:1px solid #444b46;border-radius:12px;background:#1c211fdb;box-shadow:0 16px 35px #0008;backdrop-filter:blur(20px)}.toolbtn{height:35px;display:flex;align-items:center;gap:7px;border:0;border-radius:8px;color:#b1bbb4;background:transparent;font-size:11px;font-weight:700;padding:0 13px;white-space:nowrap}.toolbtn:hover{background:#ffffff13;color:#fff}.toolbtn.active{background:#3c4736;color:var(--accent)}.toolbtn:disabled{opacity:.32}.bar-sep{height:22px;width:1px;background:#444a45;margin:0 2px}.keyboard-help{position:absolute;z-index:2;bottom:88px;left:50%;transform:translateX(-50%);font-size:10px;color:#8b9790;letter-spacing:.03em;white-space:nowrap;pointer-events:none}.kbd{border:1px solid #ffffff25;background:#ffffff0e;border-radius:4px;padding:2px 4px;font-size:9px;color:#afb7b1}
#toast{position:absolute;bottom:95px;right:30px;z-index:50;background:#2e3532;padding:12px 17px;border:1px solid #626c62;border-radius:10px;box-shadow:0 16px 50px #0008;min-width:210px;max-width:360px;font-size:11px;display:none}.modal-backdrop{position:absolute;inset:0;z-index:50;background:#090b0acb;backdrop-filter:blur(12px);display:none;align-items:center;justify-content:center;padding:20px}.modal-backdrop.open{display:flex}.modal{width:min(780px,100%);max-height:min(760px,calc(100vh - 40px));overflow:auto;background:#1a1f1e;border:1px solid #515a51;border-radius:19px;padding:26px;box-shadow:0 40px 130px #000c}.modal-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}.modal-title{font-size:23px;font-weight:630;letter-spacing:-.055em}.modal-sub{font-size:11px;color:var(--muted);margin-top:2px}.modal-close{width:35px;height:35px;border:1px solid #ffffff24;border-radius:9px;background:#2c322e;color:white;font-size:20px}.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}.gallery-card{border:1px solid #434b46;border-radius:13px;overflow:hidden;background:#222827;cursor:pointer;position:relative}.gallery-card:hover{border-color:var(--accent)}.gallery-card.selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}.gallery-cover{height:135px;position:relative;background:linear-gradient(135deg,#3a4941,#222e2b);display:flex;align-items:center;justify-content:center;color:#a4baa8}.gallery-cover img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}.gallery-content{padding:12px}.gallery-name{font-size:12px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.gallery-details{font-size:10px;color:var(--muted);margin-top:4px}.gallery-delete{position:absolute;right:8px;top:8px;border:1px solid #ffffff26;border-radius:6px;padding:5px 8px;background:#101312ce;color:#fff;font-size:10px}.gallery-empty{grid-column:1/-1;text-align:center;padding:65px 10px;color:var(--muted);font-size:12px}.foot-note{position:absolute;left:31px;bottom:25px;font-size:10px;color:#717a75;letter-spacing:.035em;z-index:2;pointer-events:none}
@media(max-width:1040px){.scene-heading{display:none}.panel.right{right:14px}.panel{left:14px}.topbar{padding:0 18px}}
@media(max-width:700px){.topbar{height:63px}.brandcaption,.statuspill,.topbtn .btn-text{display:none}.brandname{font-size:18px}.topbtn{padding:0 10px}.panel{top:74px!important;left:10px!important;width:min(308px,calc(100vw - 20px))!important;max-height:calc(100vh - 195px)}.panel.right{top:auto!important;bottom:92px;right:10px!important;left:auto!important;width:206px!important}.panel.right .panel-body{padding:11px}.bottom{bottom:13px}.toolbtn{padding:0 9px}.keyboard-help,.foot-note{display:none}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.modal{padding:17px}}
</style>
</head>
<body>
<div id="viewport"><canvas id="glCanvas"></canvas></div>
<header class="topbar">
 <div class="brand">
  <svg class="logo" viewBox="0 0 40 40" fill="none" aria-label="SPLAR logo"><path d="M20 4.7 34.1 13 34.1 27 20 35.3 5.9 27 5.9 13Z" stroke="#d8f577" stroke-width="1.4" opacity=".62"/><path d="m20 12.4 7.8 4.4v8.4L20 29.6l-7.8-4.4v-8.4Z" stroke="#d8f577" stroke-width="1.8"/><path d="m12.2 16.8 7.8 4.4 7.8-4.4M20 21.2v8.4" stroke="#d8f577" stroke-width="1.5"/><circle cx="20" cy="4.7" r="2" fill="#d8f577"/><circle cx="34.1" cy="27" r="2" fill="#d8f577"/><circle cx="5.9" cy="27" r="2" fill="#d8f577"/></svg>
  <span class="brandname">SPLAR</span><span class="brandcaption">Spatial capture</span>
 </div>
 <div class="topright"><span class="statuspill"><span class="statusdot"></span><span id="engineStatus">LOCAL ENGINE</span></span>
  <button class="topbtn" id="galleryButton"><svg class="ico" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg><span class="btn-text">Gallery</span><span id="galleryCount" class="counter">0</span></button>
  <button class="topbtn green" id="newSceneButton"><svg class="ico" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg><span class="btn-text">New room</span></button>
 </div>
</header>
<div class="scene-heading" id="sceneHeading"><div class="eyebrow" id="sceneEyebrow">INTERACTIVE 3D PREVIEW</div><h1 id="sceneTitle">The spatial canvas.</h1><p id="sceneSubtitle">Add overlapping photos to reconstruct your room.</p></div>
<section class="panel" id="capturePanel" aria-label="Capture panel">
 <div class="panel-head"><span class="panel-title"><span class="head-mark"></span>Capture workspace</span><button class="panel-toggle" title="Collapse or expand panel" aria-label="Toggle capture panel"><svg class="ico" viewBox="0 0 24 24"><path d="m6 15 6-6 6 6"/></svg></button></div>
 <div class="panel-body"><div class="section-label"><span>Source images</span><span class="counter" id="photoCount">0</span></div>
  <div class="drop" id="dropzone" role="button" tabindex="0"><div class="drop-graphic"><svg class="ico" style="width:21px;height:21px" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.4"/><path d="m4 17 6-6 4 4 2-2 5 5"/></svg></div><strong>Drop your room photos</strong><small>or click to browse · JPG, PNG, WebP</small></div>
  <div class="thumbs" id="thumbnails"><button class="thumb add" title="Add photos" id="addThumb"><span>+</span></button></div>
  <input type="file" accept="image/jpeg,image/png,image/webp" multiple id="fileInput" hidden>
  <button class="primary" id="processButton" disabled><svg class="ico" viewBox="0 0 24 24"><path d="m9 5 10 7-10 7V5Z"/></svg><span>Process room</span></button>
  <div class="process-meta"><span id="processLabel">Add at least 2 overlapping images</span><span id="processPercent">READY</span></div><div class="meter" id="meter"><div id="meterFill"></div></div>
  <div class="section-divider"></div>
  <div class="micro">Capture notes</div><div class="detail" style="margin-top:8px">Move around the room. Keep walls, furniture and corners visible across consecutive shots.</div>
 </div>
</section>
<section class="panel right" id="scenePanel" aria-label="Scene details panel">
 <div class="panel-head"><span class="panel-title"><span class="head-mark"></span>Scene inspector</span><button class="panel-toggle" title="Collapse or expand panel" aria-label="Toggle scene panel"><svg class="ico" viewBox="0 0 24 24"><path d="m6 15 6-6 6 6"/></svg></button></div>
 <div class="panel-body"><div class="micro">Reconstruction</div><div class="rowline"><span class="label">Scene</span><span class="value" id="inspectName">Demo room</span></div><div class="rowline"><span class="label">Input views</span><span class="value" id="inspectViews">—</span></div><div class="rowline"><span class="label">Gaussians</span><span class="value" id="inspectPoints">—</span></div><div class="rowline"><span class="label">Renderer</span><span class="value">WebGL 2</span></div><div class="rowline"><span class="label">Mode</span><span class="value" id="inspectMode">Demo</span></div>
  <div class="section-divider"></div><div class="section-label" style="margin:0">Splat size <span class="counter" id="sizeValue">100%</span></div><input class="range" id="splatSize" type="range" min="40" max="240" value="100"><div class="detail">Adjust point coverage for close-up views.</div>
  <button id="exportButton" class="link-btn hidden"><svg class="ico" viewBox="0 0 24 24"><path d="M12 3v13m0 0-5-5m5 5 5-5M4 18v3h16v-3"/></svg>Export PLY</button>
 </div>
</section>
<div class="keyboard-help">Drag to orbit · Scroll to zoom · Right-drag to pan · <span class="kbd">W A S D</span> to move in Fly mode</div>
<div class="bottom" role="toolbar" aria-label="3D camera controls">
 <button class="toolbtn active" id="orbitButton"><svg class="ico" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M4 12h16M12 4c4 4 4 12 0 16"/></svg>Orbit</button>
 <button class="toolbtn" id="flyButton"><svg class="ico" viewBox="0 0 24 24"><path d="m12 2 4 9 6 1-10 10L2 12l6-1Z"/></svg>Fly</button>
 <span class="bar-sep"></span><button class="toolbtn" id="fitButton"><svg class="ico" viewBox="0 0 24 24"><path d="M8 3H3v5m18 0V3h-5M3 16v5h5m8 0h5v-5"/></svg>Fit</button>
 <span class="bar-sep"></span><button class="toolbtn" id="xrButton" disabled title="Requires a WebXR-compatible headset and secure context"><svg class="ico" viewBox="0 0 24 24"><rect x="2" y="7" width="20" height="11" rx="4"/><path d="M7 12h3m4 0h3"/></svg><span id="xrLabel">XR</span></button>
</div>
<div class="foot-note">SPLAR / LOCAL-FIRST SPATIAL WORKSPACE · <span id="appVersion">v0.3.0</span></div>
<div id="toast" role="status"></div>
<div class="modal-backdrop" id="galleryModal" role="dialog" aria-modal="true" aria-label="Room gallery"><div class="modal"><div class="modal-top"><div><div class="modal-title">Your spaces<span style="color:var(--accent)">.</span></div><div class="modal-sub">Stored locally · SQLite</div></div><button class="modal-close" id="closeGallery" aria-label="Close gallery">×</button></div><div class="gallery" id="galleryGrid"></div></div></div>
<script>
'use strict';
const $ = s => document.querySelector(s);
const ICON_CLOSE = '×';
const canvas=$('#glCanvas');
const gl=canvas.getContext('webgl2',{alpha:true,antialias:false,depth:true,premultipliedAlpha:false,xrCompatible:true});
let scene=null, scenes=[], poll=null, currentDemo=true, pointCount=0, selectedId=localStorage.getItem('splar.scene')||null;
let xrSession=null, xrSpace=null, xrMode=null, splatScale=1, toastTimer=0;
function toast(message){const t=$('#toast');t.textContent=message;t.style.display='block';clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.style.display='none',4500)}
async function api(url,options={}){const res=await fetch(url,options);if(!res.ok){let msg=`HTTP ${res.status}`;try{const v=await res.json();msg=v.error||v.message||msg}catch{}throw Error(msg)}const type=res.headers.get('content-type')||'';return type.includes('application/json')?res.json():res}
function compact(n){return n>999?`${(n/1000).toFixed(n>=10000?0:1)}k`:String(n)}
function safeText(n){return String(n??'')}

// The existing tab stays on the same URL and reloads after a local code deploy.
let initialAppVersion=null;
async function pollAppVersion(){
 try{
  const response=await fetch('/api/version',{cache:'no-store'});
  if(!response.ok)return;
  const data=await response.json();
  if(!data.version)return;
  if(initialAppVersion!==null && data.version!==initialAppVersion){location.reload();return}
  initialAppVersion=data.version;
  $('#appVersion').textContent='v'+data.version;
 }catch(_){/* Flask may be restarting */}
}
pollAppVersion();
setInterval(pollAppVersion,5000);

// Custom WebGL2 renderer: camera-facing, translucent 3D Gaussian billboards.
let gpu=null, instBuffer=null, prog=null, vao=null, uniforms={};
const stride=8; // center (xyz), color (rgb), radius, opacity
let rawInstances=null, sortedInstances=null, order=null, lastSort=0, sortPending=true;
function compile(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
function setupGL(){
 if(!gl){toast('WebGL 2 is required to view scenes.');return}
 const vertex=`#version 300 es
 precision highp float;
 layout(location=0) in vec2 aCorner;
 layout(location=1) in vec3 aPosition;
 layout(location=2) in vec3 aColor;
 layout(location=3) in float aRadius;
 layout(location=4) in float aOpacity;
 uniform mat4 uView;uniform mat4 uProj;uniform float uScale;
 out vec2 vCoord;out vec3 vColor;out float vOpacity;
 void main(){vec4 viewPos=uView*vec4(aPosition,1.0);viewPos.xy+=aCorner*aRadius*uScale;
 gl_Position=uProj*viewPos;vCoord=aCorner;vColor=aColor;vOpacity=aOpacity;}`;
 const fragment=`#version 300 es
 precision highp float;
 in vec2 vCoord;in vec3 vColor;in float vOpacity;out vec4 frag;
 void main(){float alpha=exp(-3.35*dot(vCoord,vCoord))*vOpacity;
 if(alpha<0.016)discard;frag=vec4(vColor*alpha,alpha);}`;
 prog=gl.createProgram();gl.attachShader(prog,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(prog,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(prog);
 if(!gl.getProgramParameter(prog,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(prog));
 uniforms={view:gl.getUniformLocation(prog,'uView'),proj:gl.getUniformLocation(prog,'uProj'),scale:gl.getUniformLocation(prog,'uScale')};
 vao=gl.createVertexArray();gl.bindVertexArray(vao);
 const corners=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,corners);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,1,1,-1,1]),gl.STATIC_DRAW);
 gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
 const indices=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array([0,1,2,0,2,3]),gl.STATIC_DRAW);
 instBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,instBuffer);
 for(const [loc,num,offset] of [[1,3,0],[2,3,12],[3,1,24],[4,1,28]]){
  gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,num,gl.FLOAT,false,32,offset);gl.vertexAttribDivisor(loc,1);
 }
 gl.bindVertexArray(null);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.enable(gl.DEPTH_TEST);gl.depthMask(false);gl.disable(gl.CULL_FACE);
}
function eyeMat(eye,target,up=[0,1,0]){let z=norm(sub(eye,target)),x=norm(cross(up,z)),y=cross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1])}
function perspective(fovy,aspect,near,far){let f=1/Math.tan(fovy/2),a=1/(near-far);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*a,-1,0,0,2*far*near*a,0])}
function add(a,b){return a.map((x,i)=>x+b[i])}function sub(a,b){return a.map((x,i)=>x-b[i])}function mul(v,k){return v.map(x=>x*k)}function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function norm(a){let l=Math.hypot(...a)||1;return mul(a,1/l)}
let cam={target:[0,.1,0],theta:.42,phi:1.31,radius:10,pos:[0,0,10],yaw:0,pitch:0,mode:'orbit'};
let drag=null, downKeys=new Set(), autoMoveAt=0;
function orbitPos(){let p=cam.phi;return add(cam.target,[cam.radius*Math.sin(p)*Math.sin(cam.theta),cam.radius*Math.cos(p),cam.radius*Math.sin(p)*Math.cos(cam.theta)])}
function flyDirection(){return [Math.sin(cam.yaw)*Math.cos(cam.pitch),Math.sin(cam.pitch),-Math.cos(cam.yaw)*Math.cos(cam.pitch)]}
function viewMatrix(){return cam.mode==='orbit'?eyeMat(orbitPos(),cam.target):eyeMat(cam.pos,add(cam.pos,flyDirection()))}
function fit(){cam.target=[0,0,0];cam.theta=.42;cam.phi=1.34;cam.radius=11;cam.mode='orbit';setCameraButtons();sortPending=true}
function setCameraButtons(){$('#orbitButton').classList.toggle('active',cam.mode==='orbit');$('#flyButton').classList.toggle('active',cam.mode==='fly')}
function cameraMode(m){if(m===cam.mode)return;if(m==='fly'){cam.pos=orbitPos();let d=norm(sub(cam.target,cam.pos));cam.yaw=Math.atan2(d[0],-d[2]);cam.pitch=Math.asin(Math.max(-1,Math.min(1,d[1])))}else{cam.target=add(cam.pos,mul(flyDirection(),cam.radius));cam.phi=Math.max(.1,Math.min(Math.PI-.1,Math.acos((cam.pos[1]-cam.target[1])/cam.radius)));cam.theta=Math.atan2(cam.pos[0]-cam.target[0],cam.pos[2]-cam.target[2])}cam.mode=m;setCameraButtons();sortPending=true}
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{if(xrSession)return;canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,mode:(e.button===2||e.shiftKey)?'pan':'turn'};canvas.classList.add('dragging')});
canvas.addEventListener('pointermove',e=>{if(!drag)return;let dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
 if(cam.mode==='fly'){cam.yaw-=dx*.005;cam.pitch=Math.max(-1.49,Math.min(1.49,cam.pitch-dy*.005))}
 else if(drag.mode==='turn'){cam.theta-=dx*.005;cam.phi=Math.max(.085,Math.min(Math.PI-.085,cam.phi-dy*.005))}
 else{let speed=cam.radius*.0013;let front=norm(sub(cam.target,orbitPos())),right=norm(cross(front,[0,1,0]));cam.target=add(cam.target,add(mul(right,-dx*speed),[0,dy*speed,0]))}sortPending=true});
function stopDrag(){drag=null;canvas.classList.remove('dragging')}canvas.addEventListener('pointerup',stopDrag);canvas.addEventListener('pointercancel',stopDrag);
canvas.addEventListener('wheel',e=>{e.preventDefault();if(cam.mode==='fly'){cam.pos=add(cam.pos,mul(flyDirection(),e.deltaY>0?-.32:.32))}else{cam.radius=Math.max(.25,Math.min(50,cam.radius*Math.exp(e.deltaY*.001)))}sortPending=true},{passive:false});
window.addEventListener('keydown',e=>{if(e.target.closest('input,button,textarea'))return;if('wasdqeshift'.includes(e.key.toLowerCase())&&e.key.length===1)downKeys.add(e.key.toLowerCase());if(e.key==='Shift')downKeys.add('shift')});
window.addEventListener('keyup',e=>downKeys.delete(e.key.toLowerCase()));
window.addEventListener('blur',()=>downKeys.clear());
function advance(dt){if(cam.mode!=='fly'||!downKeys.size)return;let f=flyDirection(),flat=norm([f[0],0,f[2]]),r=[-flat[2],0,flat[0]];let delta=[0,0,0];if(downKeys.has('w'))delta=add(delta,flat);if(downKeys.has('s'))delta=sub(delta,flat);if(downKeys.has('d'))delta=add(delta,r);if(downKeys.has('a'))delta=sub(delta,r);if(downKeys.has('e'))delta[1]+=1;if(downKeys.has('q'))delta[1]-=1;if(Math.hypot(...delta)){cam.pos=add(cam.pos,mul(norm(delta),Math.min(.1,dt/1000)* (downKeys.has('shift')?8:3)));sortPending=true}}
function uploadSplat(arrayBuffer){const dv=new DataView(arrayBuffer),decoder=new TextDecoder();const magic=decoder.decode(new Uint8Array(arrayBuffer,0,8));if(magic!=='SPLAR01\0')throw Error('Unsupported SPLAR file');const count=dv.getUint32(8,true);if(arrayBuffer.byteLength!==12+count*20)throw Error('Corrupt splat file');
 const a=new Float32Array(count*stride);
 for(let i=0;i<count;i++){const src=12+i*20,d=i*stride;a[d]=dv.getFloat32(src,true);a[d+1]=dv.getFloat32(src+4,true);a[d+2]=dv.getFloat32(src+8,true);a[d+3]=dv.getUint8(src+16)/255;a[d+4]=dv.getUint8(src+17)/255;a[d+5]=dv.getUint8(src+18)/255;a[d+6]=dv.getFloat32(src+12,true);a[d+7]=dv.getUint8(src+19)/255}
 rawInstances=a;sortedInstances=new Float32Array(a.length);order=Array.from({length:count},(_,i)=>i);pointCount=count;sortPending=true;lastSort=0;
 gl.bindBuffer(gl.ARRAY_BUFFER,instBuffer);gl.bufferData(gl.ARRAY_BUFFER,a,gl.DYNAMIC_DRAW);gl.bindBuffer(gl.ARRAY_BUFFER,null);fit();return count}
async function showSplat(url){if(!gl)return;try{const response=await fetch(url);if(!response.ok)throw Error('Scene file unavailable');uploadSplat(await response.arrayBuffer());$('#inspectPoints').textContent=pointCount.toLocaleString()}catch(err){toast(err.message)}}
function sortSplat(v,t){if(!rawInstances||!sortPending||t-lastSort<300)return;lastSort=t;sortPending=false;
 // Back-to-front sorting matters for translucent Gaussian blending.
 const a=rawInstances;order.sort((i,j)=>{
  let k=i*stride,l=j*stride;return (v[2]*a[k]+v[6]*a[k+1]+v[10]*a[k+2]+v[14])-(v[2]*a[l]+v[6]*a[l+1]+v[10]*a[l+2]+v[14])});
 for(let j=0;j<order.length;j++)sortedInstances.set(a.subarray(order[j]*stride,order[j]*stride+stride),j*stride);
 gl.bindBuffer(gl.ARRAY_BUFFER,instBuffer);gl.bufferSubData(gl.ARRAY_BUFFER,0,sortedInstances);
}
function paint(view,projection,time){if(!prog||!pointCount)return;sortSplat(view,time);gl.useProgram(prog);gl.bindVertexArray(vao);gl.uniformMatrix4fv(uniforms.view,false,view);gl.uniformMatrix4fv(uniforms.proj,false,projection);gl.uniform1f(uniforms.scale,splatScale);gl.drawElementsInstanced(gl.TRIANGLES,6,gl.UNSIGNED_SHORT,0,pointCount);gl.bindVertexArray(null)}
let previous=0;
function draw(time){requestAnimationFrame(draw);if(!gl||xrSession)return;const dt=Math.max(0,time-previous);previous=time;advance(dt);let pixelRatio=Math.min(2,window.devicePixelRatio||1),w=Math.floor(canvas.clientWidth*pixelRatio),h=Math.floor(canvas.clientHeight*pixelRatio);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,w,h);gl.clearColor(.038,.047,.047,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);paint(viewMatrix(),perspective(Math.PI/3,w/Math.max(1,h),.05,300),time)}
async function checkXR(){if(!navigator.xr||!gl)return;let mode=null;try{if(await navigator.xr.isSessionSupported('immersive-vr'))mode='immersive-vr';else if(await navigator.xr.isSessionSupported('immersive-ar'))mode='immersive-ar'}catch{}if(mode){xrMode=mode;$('#xrButton').disabled=false;$('#xrLabel').textContent=mode==='immersive-vr'?'Enter VR':'Enter AR'}}
async function enterXR(){if(xrSession){await xrSession.end();return}if(!xrMode){toast('WebXR unavailable in this browser.');return}try{
 await gl.makeXRCompatible();const session=await navigator.xr.requestSession(xrMode,{optionalFeatures:['local-floor','bounded-floor']});xrSession=session;
 session.updateRenderState({baseLayer:new XRWebGLLayer(session,gl)});xrSpace=await session.requestReferenceSpace('local-floor').catch(()=>session.requestReferenceSpace('local'));
 $('#xrLabel').textContent='Exit XR';session.addEventListener('end',()=>{xrSession=null;xrSpace=null;$('#xrLabel').textContent=xrMode==='immersive-vr'?'Enter VR':'Enter AR';sortPending=true});session.requestAnimationFrame(onXRFrame)
 }catch(err){toast(`XR: ${err.message}`);xrSession=null}}
function onXRFrame(time,frame){if(!xrSession||!gl)return;const sess=frame.session;sess.requestAnimationFrame(onXRFrame);const pose=frame.getViewerPose(xrSpace);if(!pose)return;const layer=sess.renderState.baseLayer;gl.bindFramebuffer(gl.FRAMEBUFFER,layer.framebuffer);gl.clearColor(0,0,0,xrMode==='immersive-ar'?0:1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);for(const view of pose.views){const vp=layer.getViewport(view);gl.viewport(vp.x,vp.y,vp.width,vp.height);paint(view.transform.inverse.matrix,view.projectionMatrix,time)}}
$('#orbitButton').onclick=()=>cameraMode('orbit');$('#flyButton').onclick=()=>cameraMode('fly');$('#fitButton').onclick=fit;$('#xrButton').onclick=enterXR;
$('#splatSize').oninput=e=>{splatScale=Number(e.target.value)/100;$('#sizeValue').textContent=e.target.value+'%'};

// API and persistence.
async function refreshScenes(){scenes=await api('/api/scenes');$('#galleryCount').textContent=scenes.length}
async function setScene(s){scene=s;selectedId=s?.id||null;if(selectedId)localStorage.setItem('splar.scene',selectedId);else localStorage.removeItem('splar.scene');currentDemo=!s||s.status!=='ready';
 $('#photoCount').textContent=String(s?.images?.length||0);$('#inspectName').textContent=s?.title||'Demo room';$('#inspectViews').textContent=s?String(s.images.length):'—';$('#inspectMode').textContent=s?(s.status==='ready'?'Reconstructed':s.status==='processing'?'Processing':'Draft'):'Demo';
 $('#sceneTitle').textContent=s?.title||'The spatial canvas.';$('#sceneEyebrow').textContent=s?.status==='ready'?'RECONSTRUCTED 3D SCENE':s?.status==='processing'?'GENERATING GEOMETRY':'INTERACTIVE 3D PREVIEW';
 $('#sceneSubtitle').textContent=s?.status==='ready'?'Navigate your reconstructed space.':s?.status==='processing'?'Local geometry engine is running.':'Add overlapping photos to reconstruct your room.';
 $('#sceneHeading').classList.toggle('condensed',!!s);
 const count=s?.images?.length||0;$('#processButton').disabled=count<2||s?.status==='processing';$('#processButton span').textContent=s?.status==='processing'?'Processing…':s?.status==='ready'?'Reprocess room':'Process room';
 $('#processLabel').textContent=s?.status==='error'?(s.error||'Processing failed'):s?.status==='processing'?s.stage:s?.status==='ready'?'3D scene is ready':count<2?'Add at least 2 overlapping images':`${count} photos ready`;
 $('#processLabel').title=s?.error||'';$('#processPercent').textContent=s?.status==='processing'?'WORKING':s?.status==='error'?'ERROR':s?.status==='ready'?'DONE':'READY';$('#meter').classList.toggle('loading',s?.status==='processing');$('#meterFill').style.width=s?.status==='ready'?'100%':'0%';
 $('#exportButton').classList.toggle('hidden',s?.status!=='ready');$('#exportButton').onclick=()=>{if(s?.ply_url)window.location.href=s.ply_url};
 const t=$('#thumbnails');t.replaceChildren();for(const img of s?.images||[]){const box=document.createElement('div');box.className='thumb';const image=document.createElement('img');image.src=img.url;image.alt=img.filename;image.loading='lazy';image.decoding='async';box.append(image);if(s.status!=='processing'){const remove=document.createElement('button');remove.className='remove';remove.textContent='×';remove.title='Remove '+img.filename;remove.onclick=async()=>{try{const updated=await api(`/api/scenes/${s.id}/images/${img.id}`,{method:'DELETE'});await setScene(updated);await showSplat('/api/demo/splat')}catch(e){toast(e.message)}};box.append(remove)}t.append(box)}
 const add=document.createElement('button');add.className='thumb add';add.title='Add more photos';add.innerHTML='<span>+</span>';add.onclick=()=>$('#fileInput').click();if(s?.status!=='processing')t.append(add);
 if(s?.status==='ready'&&s.splat_url){await showSplat(s.splat_url)}else{await showSplat('/api/demo/splat')}
 if(s?.status==='processing')startPolling();else stopPolling();
}
async function createScene(){try{const s=await api('/api/scenes',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await refreshScenes();await setScene(s);toast('New room created')}catch(err){toast(err.message)}}
let isUploading=false;
async function addFiles(files){
 if(isUploading||!files?.length)return;
 isUploading=true;
 try{
  if(!scene||scene.status==='processing'){
   scene=await api('/api/scenes',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  }
  const arr=Array.from(files);
  // There is no maximum number of photos per room. Split uploads into
  // requests below Flask's 80 MB request limit, regardless of gallery size.
  const maxBatchBytes=48*1024*1024;
  const batches=[];let batch=[],bytes=0;
  for(const file of arr){
   if(file.size>16*1024*1024)throw Error(`${file.name}: each image must be 16 MB or smaller.`);
   if(batch.length&&(bytes+file.size>maxBatchBytes||batch.length>=20)){
    batches.push(batch);batch=[];bytes=0;
   }
   batch.push(file);bytes+=file.size;
  }
  if(batch.length)batches.push(batch);
  let uploaded=0;
  for(const group of batches){
   $('#processLabel').textContent=`Uploading photos ${uploaded+1}–${uploaded+group.length} of ${arr.length}…`;
   const data=new FormData();group.forEach(f=>data.append('files',f));
   scene=await api(`/api/scenes/${scene.id}/images`,{method:'POST',body:data});
   uploaded+=group.length;
   $('#photoCount').textContent=String(scene.images.length);
  }
  await refreshScenes();await setScene(scene);
  toast(`${uploaded} photo${uploaded===1?'':'s'} added`);
 }catch(err){
  toast(err.message);
  if(scene?.id){try{await setScene(await api(`/api/scenes/${scene.id}`))}catch(_){}}
 }finally{isUploading=false;$('#fileInput').value=''}
}
$('#fileInput').onchange=e=>addFiles(e.target.files);$('#dropzone').onclick=()=>$('#fileInput').click();$('#dropzone').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#fileInput').click()}};
for(const ev of ['dragenter','dragover'])$('#dropzone').addEventListener(ev,e=>{e.preventDefault();$('#dropzone').classList.add('dragover')});
for(const ev of ['dragleave','drop'])$('#dropzone').addEventListener(ev,e=>{e.preventDefault();$('#dropzone').classList.remove('dragover')});
$('#dropzone').addEventListener('drop',e=>addFiles(e.dataTransfer.files));
$('#newSceneButton').onclick=createScene;
$('#processButton').onclick=async()=>{if(!scene)return;try{const s=await api(`/api/scenes/${scene.id}/process`,{method:'POST'});await setScene(s)}catch(err){toast(err.message)}};
function stopPolling(){if(poll){clearInterval(poll);poll=null}}
function startPolling(){if(poll||!scene)return;poll=setInterval(async()=>{if(!scene||scene.status!=='processing'){stopPolling();return}try{const s=await api(`/api/scenes/${scene.id}`);if(s.status!==scene.status){await setScene(s);if(s.status==='error')toast(s.error||'Reconstruction failed');if(s.status==='ready')toast('Room reconstruction ready')}else{$('#processLabel').textContent=s.stage;scene=s}}catch(err){console.warn(err)}},1600)}
async function openGallery(){try{await refreshScenes();const grid=$('#galleryGrid');grid.replaceChildren();if(!scenes.length){grid.innerHTML='<div class="gallery-empty">No saved rooms yet. Add your first photos.</div>'}
 for(const s of scenes){const card=document.createElement('div');card.className='gallery-card'+(s.id===selectedId?' selected':'');const cover=document.createElement('div');cover.className='gallery-cover';if(s.images.length){const img=document.createElement('img');img.src=s.images[0].url;cover.append(img)}else{cover.textContent='◈'};
 const content=document.createElement('div');content.className='gallery-content';const name=document.createElement('div');name.className='gallery-name';name.textContent=s.title;const meta=document.createElement('div');meta.className='gallery-details';meta.textContent=`${s.images.length} views · ${s.status==='ready'?compact(s.point_count)+' splats':s.status}`;content.append(name,meta);const del=document.createElement('button');del.className='gallery-delete';del.textContent='Delete';del.title='Permanently delete this room';del.onclick=async e=>{e.stopPropagation();if(!confirm('Delete this room and its photos?'))return;try{await api(`/api/scenes/${s.id}`,{method:'DELETE'});if(s.id===selectedId){scene=null;selectedId=null;await setScene(null)}await openGallery()}catch(err){toast(err.message)}};
 card.onclick=async()=>{stopPolling();await setScene(await api(`/api/scenes/${s.id}`));closeGallery()};card.append(cover,content,del);grid.append(card)}
 $('#galleryModal').classList.add('open')}catch(err){toast(err.message)}}
function closeGallery(){$('#galleryModal').classList.remove('open')}
$('#galleryButton').onclick=openGallery;$('#closeGallery').onclick=closeGallery;$('#galleryModal').addEventListener('click',e=>{if(e.target===$('#galleryModal'))closeGallery()});document.addEventListener('keydown',e=>{if(e.key==='Escape')closeGallery()});

// Movable and collapsible inspector windows.
for(const panel of document.querySelectorAll('.panel')){
 const head=panel.querySelector('.panel-head');const toggle=panel.querySelector('.panel-toggle');
 toggle.onclick=()=>{panel.classList.toggle('collapsed');const p=toggle.querySelector('path');p.setAttribute('d',panel.classList.contains('collapsed')?'m6 9 6 6 6-6':'m6 15 6-6 6 6')};
 let move=null;head.addEventListener('pointerdown',e=>{if(e.target.closest('button')||window.innerWidth<700)return;const rect=panel.getBoundingClientRect();move={dx:e.clientX-rect.left,dy:e.clientY-rect.top};head.setPointerCapture(e.pointerId)});
 head.addEventListener('pointermove',e=>{if(!move)return;const maxX=window.innerWidth-panel.offsetWidth-6,maxY=window.innerHeight-panel.querySelector('.panel-head').offsetHeight-6;panel.style.right='auto';panel.style.left=Math.max(6,Math.min(maxX,e.clientX-move.dx))+'px';panel.style.top=Math.max(6,Math.min(maxY,e.clientY-move.dy))+'px'});
 head.addEventListener('pointerup',()=>{move=null});head.addEventListener('pointercancel',()=>{move=null});
}
async function init(){try{if(gl){setupGL();requestAnimationFrame(draw)}await showSplat('/api/demo/splat');$('#inspectViews').textContent='Demo';$('#inspectMode').textContent='Demo';await refreshScenes();if(selectedId){const stored=scenes.find(s=>s.id===selectedId);if(stored)await setScene(stored)}checkXR()}catch(err){console.error(err);toast('Startup error: '+err.message)}}
init();
</script>
</body>
</html>'''

if __name__ == '__main__':
    init_db()
    make_demo()
    host = os.environ.get('SPLAR_BIND', '0.0.0.0')
    port = int(os.environ.get('SPLAR_PORT', '5463'))
    ssl_context = None
    if os.environ.get('SPLAR_CERT') and os.environ.get('SPLAR_KEY'):
        ssl_context = (os.environ['SPLAR_CERT'], os.environ['SPLAR_KEY'])
    elif os.environ.get('SPLAR_HTTPS') == '1':
        ssl_context = 'adhoc'  # development only; use a trusted certificate for headset access.
    protocol = 'https' if ssl_context else 'http'
    print(f'\n  SPLAR  /  {protocol}://127.0.0.1:{port} (bind {host})  /  SQLite: {DATABASE}\n', flush=True)
    app.run(host=host, port=port, debug=False, threaded=True, ssl_context=ssl_context)
