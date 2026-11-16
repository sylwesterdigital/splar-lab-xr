#!/usr/bin/env bash
# SPLAR — local release watcher. Requires only macOS bash + Python 3 stdlib.
# Usage: ./watch-splar.sh [--once]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u - "$REPO_ROOT" "$@" <<'PYTHON'
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import shutil
import signal
import ssl
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime

ROOT = Path(sys.argv[1]).resolve()
ONCE = '--once' in sys.argv[2:]
CORE = ROOT / 'core'
UPDATES = ROOT / 'updates'
STATE = UPDATES / '.watcher'
VERSION_FILE = ROOT / 'VERSION'
PID_FILE = STATE / 'flask.pid'
PENDING_FILE = STATE / 'pending.json'
LOG_FILE = STATE / 'flask.log'
PORT = int(os.getenv('SPLAR_PORT', '5463'))
BIND = os.getenv('SPLAR_BIND', '0.0.0.0')
POLL = max(2, int(os.getenv('SPLAR_WATCH_INTERVAL', '4')))
URL = ('https' if os.getenv('SPLAR_HTTPS') == '1' or os.getenv('SPLAR_CERT') else 'http') + f'://127.0.0.1:{PORT}'
VERSION_RE = re.compile(r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$')
PY_SOURCE = 'core/splar.py'
last_notice = {}
last_publish_attempt = 0.0
COLOR = sys.stdout.isatty() and 'NO_COLOR' not in os.environ and os.getenv('TERM') != 'dumb'
ANSI = {'info':'\033[38;5;117m','scan':'\033[38;5;244m','ok':'\033[38;5;120m',
        'warn':'\033[38;5;214m','error':'\033[38;5;203m','deploy':'\033[38;5;183m'}
RESET = '\033[0m' if COLOR else ''


def color(text, style):
    return f'{ANSI.get(style, ANSI["info"])}{text}{RESET}' if COLOR else text


last_output_at = 0.0


def log(message, style='info'):
    global last_output_at
    moment = time.monotonic()
    # Separate real events visually; do not print anything during idle scans.
    if last_output_at and moment - last_output_at > 1.0:
        print(flush=True)
    clock = datetime.now().astimezone().strftime('%H:%M:%S')
    print(f'{color("[" + clock + "]", "scan")} {color("SPLAR", style)} | {message}', flush=True)
    last_output_at = moment


def local_ip():
    # macOS: prefer active Wi-Fi / Ethernet interfaces without an external network request.
    if sys.platform == 'darwin':
        for interface in ('en0', 'en1', 'en2', 'en3', 'en4', 'en5', 'en6', 'en7'):
            try:
                value = subprocess.run(['ipconfig', 'getifaddr', interface], text=True,
                                       capture_output=True, timeout=1).stdout.strip()
                if value and not value.startswith('127.'):
                    return value
            except (OSError, subprocess.TimeoutExpired):
                pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(('192.0.2.1', 80))  # local route selection; sends no packets
            value = sock.getsockname()[0]
            return value if value and not value.startswith('127.') else None
    except OSError:
        return None


def print_urls():
    log('Local   ' + color(URL, 'ok'), 'ok')
    addr = local_ip()
    if addr and BIND not in ('127.0.0.1', 'localhost', '::1'):
        scheme = URL.split(':',1)[0]
        log('Network ' + color(f'{scheme}://{addr}:{PORT}', 'ok'), 'ok')
    elif BIND in ('127.0.0.1', 'localhost', '::1'):
        log('Network access disabled: set SPLAR_BIND=0.0.0.0', 'warn')
    else:
        log('Network IP not detected; check your Mac LAN address', 'warn')


def notice(key, message, seconds=35):
    if time.time() - last_notice.get(key, 0) >= seconds:
        log(message, 'error' if key.startswith(('error','loop','flask','startup')) else 'warn')
        last_notice[key] = time.time()


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2) + '\n').encode())


def version_tuple(ver):
    if not VERSION_RE.fullmatch(ver):
        raise ValueError(f'Invalid semantic version: {ver!r}')
    return tuple(int(v) for v in ver.split('.'))


def current_version():
    return VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else '0.1.0'


def digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


def unpack_release(path: Path):
    """Read only approved paths; reject traversal, links, oversize and corruption."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members if not m.is_dir()]
        if len(names) != len(set(names)) or set(names) != {'release.json', PY_SOURCE}:
            raise ValueError('ZIP must contain exactly release.json and core/splar.py')
        for entry in members:
            if entry.filename.startswith('/') or '..' in Path(entry.filename).parts or '\\' in entry.filename:
                raise ValueError('Unsafe ZIP member path')
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Symlinks in updates are not allowed')
            if entry.file_size > 2 * 1024 * 1024:
                raise ValueError('An update file exceeds 2 MB')
        if archive.testzip() is not None:
            raise ValueError('ZIP checksum failed')
        manifest = json.loads(archive.read('release.json'))
        if manifest.get('app') != 'SPLAR' or manifest.get('format') != 1:
            raise ValueError('Not a SPLAR update package')
        version = manifest.get('version', '')
        version_tuple(version)
        expected = manifest.get('sha256', {}).get(PY_SOURCE)
        code = archive.read(PY_SOURCE)
        if expected != digest(code):
            raise ValueError('core/splar.py SHA-256 does not match release.json')
        compile(code.decode('utf-8'), PY_SOURCE, 'exec')
        return version, code


def git(*args, check=False):
    return subprocess.run(['git', '-C', str(ROOT), *args], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'},
                          timeout=60, check=check)


def health(base_url=None):
    url = base_url or URL
    try:
        context = ssl._create_unverified_context() if url.startswith('https:') else None
        with urllib.request.urlopen(url + '/api/health', timeout=1.7, context=context) as r:
            body = json.load(r)
        return body.get('database') == 'SQLite' and 'VGGT' in body.get('engine', '')
    except Exception:
        return False


def busy(base_url=None):
    """Avoid killing Flask while it is reconstructing a room."""
    url = base_url or URL
    if not health(url):
        return False
    try:
        context = ssl._create_unverified_context() if url.startswith('https:') else None
        with urllib.request.urlopen(url + '/api/scenes', timeout=3, context=context) as r:
            scenes = json.load(r)
        return any(item.get('status') == 'processing' for item in scenes)
    except Exception:
        # Unknown state: don't interrupt a running process.
        return True


def pid_command(pid):
    p = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ''


def process_alive(pid):
    if not pid:
        return False
    status = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True)
    return status.returncode == 0 and not status.stdout.lstrip().startswith('Z')


def listening_pids(port=None):
    if not shutil.which('lsof'):
        return []
    result = subprocess.run(['lsof', '-nP', '-t', f'-iTCP:{port or PORT}', '-sTCP:LISTEN'],
                            capture_output=True, text=True)
    return sorted({int(v) for v in result.stdout.split() if v.isdigit()})


def recorded_pid():
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def stop_server(port=None):
    port = port or PORT
    pids = listening_pids(port)
    own_pid = recorded_pid()
    if own_pid and process_alive(own_pid) and own_pid not in pids:
        # The tracked process may belong to the old listener when migrating ports.
        pids.append(own_pid)
    # Do not terminate another app occupying the port.
    for pid in pids:
        command = pid_command(pid)
        if command and 'splar.py' not in command:
            raise RuntimeError(f'Port {port} belongs to another process ({pid}: {command})')
    for pid in pids:
        if process_alive(pid):
            log(f'Stopping Flask process {pid}')
            os.kill(pid, signal.SIGTERM)
    end = time.monotonic() + 12
    while time.monotonic() < end:
        if all(not process_alive(pid) for pid in pids):
            break
        time.sleep(.25)
    for pid in pids:
        if process_alive(pid):
            raise RuntimeError(f'Flask process {pid} did not stop; intervention required')
    PID_FILE.unlink(missing_ok=True)


def python_interpreter():
    for interpreter in (CORE / '.venv/bin/python', ROOT / '.venv/bin/python'):
        if interpreter.is_file():
            return interpreter
    raise RuntimeError('Existing .venv not found. Start the original app once via core/run.zsh. Watcher never runs pip.')


def start_server():
    if health():
        return
    # A port occupied by an unknown app must not be reused or killed.
    pids = listening_pids()
    if pids:
        raise RuntimeError(f'Port {PORT} already occupied by process(es) {pids}')
    if recorded_pid() and process_alive(recorded_pid()):
        raise RuntimeError('A tracked Flask process is still running but not responding')
    py = python_interpreter()
    env = dict(os.environ)
    env['SPLAR_PORT'] = str(PORT)
    env['SPLAR_BIND'] = BIND
    env.setdefault('HF_HUB_DISABLE_XET', '1')
    env.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    with LOG_FILE.open('ab', buffering=0) as output:
        proc = subprocess.Popen([str(py), '-u', 'splar.py'], cwd=str(CORE),
                                env=env, stdin=subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True)
    PID_FILE.write_text(f'{proc.pid}\n')
    log(f'Starting existing Python environment (PID {proc.pid}, port {PORT}, bind {BIND})', 'deploy')
    until = time.monotonic() + 35
    while time.monotonic() < until:
        if health():
            log(f'Flask ready: {URL}', 'ok')
            print_urls()
            return
        if proc.poll() is not None:
            raise RuntimeError(f'Flask exited with status {proc.returncode}; inspect {LOG_FILE}')
        time.sleep(.5)
    raise RuntimeError(f'Flask health check timed out; inspect {LOG_FILE}')


def migrate_legacy_port():
    """Retire the old tracked 5050 SPLAR server before starting on 5463."""
    if PORT == 5050:
        return True
    legacy_url = 'http://127.0.0.1:5050'
    old_pid = recorded_pid()
    old_listener_pids = listening_pids(5050)
    if old_pid and process_alive(old_pid) and old_pid not in old_listener_pids:
        # When lsof is unavailable, only consider the tracked PID a legacy server
        # if port 5050 responds as SPLAR (never confuse the new 5463 server).
        if health(legacy_url) and 'splar.py' in pid_command(old_pid):
            old_listener_pids.append(old_pid)
    if not old_listener_pids:
        return True
    for pid in old_listener_pids:
        command = pid_command(pid)
        if 'splar.py' not in command:
            log(f'Port 5050 in use by another app (PID {pid}); leaving it untouched', 'warn')
            return True
    if not health(legacy_url):
        log('Existing 5050 process did not respond to SPLAR health check; leaving it untouched', 'warn')
        return True
    if busy(legacy_url):
        notice('migration', 'Reconstruction still running on 5050; defer migration to 5463')
        return False
    log('Moving existing SPLAR server from 5050 to 5463', 'deploy')
    stop_server(5050)
    return True


def restart():
    stop_server()
    start_server()


def apply_release(path, archive_hash, version, code):
    if not migrate_legacy_port():
        return False
    if busy():
        notice('busy', f'Room reconstruction active; deferring update {version}')
        return False
    backup_dir = STATE / 'backups' / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-v{current_version()}-to-v{version}'
    backup_dir.mkdir(parents=True, exist_ok=True)
    original_code = (CORE / 'splar.py').read_bytes()
    previous_version = VERSION_FILE.read_bytes() if VERSION_FILE.exists() else None
    (backup_dir / 'splar.py').write_bytes(original_code)
    if previous_version is not None:
        (backup_dir / 'VERSION').write_bytes(previous_version)
    log(f'Applying v{version} (backup: {backup_dir.relative_to(ROOT)})', 'deploy')
    try:
        atomic_bytes(CORE / 'splar.py', code)
        atomic_bytes(VERSION_FILE, (version + '\n').encode())
        restart()
    except Exception as exc:
        log(f'Update failed ({exc}); restoring old application', 'error')
        atomic_bytes(CORE / 'splar.py', original_code)
        if previous_version is None:
            VERSION_FILE.unlink(missing_ok=True)
        else:
            atomic_bytes(VERSION_FILE, previous_version)
        try:
            restart()
        except Exception as rollback_error:
            log(f'Rollback server restart failed: {rollback_error}')
        raise
    write_json(PENDING_FILE, {'version': version, 'archive': path.name, 'sha256': archive_hash})
    log(f'v{version} deployed locally; Python packages, model weights and splar_data untouched', 'ok')
    return True


def archive_release(path, version, archive_hash, folder='.applied'):
    dest_dir = UPDATES / folder
    dest_dir.mkdir(exist_ok=True)
    target = dest_dir / f'v{version}-{archive_hash[:12]}.zip'
    if path.exists():
        os.replace(path, target)
    log(f'Archived ZIP: {target.relative_to(ROOT)}')


def publish(pending):
    version = pending['version']
    if not (ROOT / '.git').exists():
        notice('git', 'No Git repository found; local update applied, GitHub push pending')
        return False
    staged = git('diff', '--cached', '--name-only').stdout.splitlines()
    if set(staged) - {'VERSION', PY_SOURCE, 'watch-splar.sh'}:
        notice('staged', 'Unrelated changes are already staged. Commit/unstage them first; SPLAR will not include them.')
        return False
    branch = git('symbolic-ref', '--short', 'HEAD')
    if branch.returncode:
        notice('branch', 'Detached HEAD: checkout main before publishing')
        return False
    if git('remote', 'get-url', 'origin').returncode:
        notice('remote', 'No origin remote: set GitHub origin to enable automatic push')
        return False
    tracked_private = git('ls-files', '--', 'core/splar_data', 'core/.venv').stdout.strip()
    if tracked_private:
        notice('private', 'Private splar_data or .venv files are already Git-tracked; refusing automatic push until untracked')
        return False
    current = current_version()
    if current != version:
        notice('version', f'Version moved to v{current}; refusing to publish stale v{version}')
        return False
    stage = git('add', '--', 'VERSION', PY_SOURCE, 'watch-splar.sh')
    if stage.returncode:
        notice('git-add', f'Git staging failed: {stage.stderr.strip()}')
        return False
    if git('diff', '--cached', '--quiet').returncode != 0:
        commit = git('commit', '-m', f'chore(splar): release v{version}')
        if commit.returncode:
            notice('commit', f'Git commit failed: {commit.stderr.strip()}')
            return False
        log(f'Committed v{version} ({git("rev-parse", "--short", "HEAD").stdout.strip()})')
    push = git('push', 'origin', f'HEAD:refs/heads/{branch.stdout.strip()}')
    if push.returncode:
        notice('push', f'Git push pending: {push.stderr.strip()}')
        return False
    log(f'Pushed v{version} to origin/{branch.stdout.strip()}', 'ok')
    return True


def process_pending():
    global last_publish_attempt
    if not PENDING_FILE.exists():
        return False
    if time.time() - last_publish_attempt < 30:
        return True
    last_publish_attempt = time.time()
    pending = json.loads(PENDING_FILE.read_text())
    if publish(pending):
        archive = UPDATES / pending['archive']
        archive_release(archive, pending['version'], pending['sha256'])
        PENDING_FILE.unlink(missing_ok=True)
    return True


def examine_zip(path):
    archive_hash = digest(path.read_bytes())
    version, code = unpack_release(path)
    installed = current_version()
    if version_tuple(version) < version_tuple(installed):
        log(f'Skipping outdated v{version}; current version is v{installed}')
        archive_release(path, version, archive_hash, '.skipped')
        return
    if version == installed:
        if digest((CORE / 'splar.py').read_bytes()) == digest(code):
            log(f'v{version} already installed; publishing/archiving this package')
            write_json(PENDING_FILE, {'version': version, 'archive': path.name, 'sha256': archive_hash})
        else:
            notice(str(path), f'v{version} conflicts with installed code; only newer versions can overwrite it')
        return
    apply_release(path, archive_hash, version, code)


def watcher_iteration(stability):
    if process_pending():
        return
    for path in sorted(UPDATES.glob('*.zip')):
        try:
            signature = (path.stat().st_size, path.stat().st_mtime_ns)
            if stability.get(path.name) != signature:
                stability[path.name] = signature
                continue  # A browser may still be writing this ZIP.
            examine_zip(path)
        except (ValueError, OSError, zipfile.BadZipFile, RuntimeError, SyntaxError, UnicodeError, json.JSONDecodeError) as exc:
            notice('error-'+path.name, f'Update {path.name} not applied: {exc}', 30)
        if PENDING_FILE.exists():
            process_pending()
            break


def main():
    if not (CORE / 'splar.py').is_file():
        raise RuntimeError('Expected core/splar.py next to the updates/ directory')
    UPDATES.mkdir(exist_ok=True)
    STATE.mkdir(exist_ok=True)
    lock = (STATE / 'watcher.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another SPLAR watcher is already running')
    version_tuple(current_version())
    if not VERSION_FILE.exists():
        atomic_bytes(VERSION_FILE, b'0.1.0\n')
    log(f'Watching {UPDATES} | current v{current_version()} | interval {POLL}s (quiet)', 'deploy')
    log('Only updates, errors and restarts are printed; no dependency or model reinstall')
    try:
        if migrate_legacy_port():
            if not health():
                start_server()
            else:
                print_urls()
    except Exception as exc:
        notice('startup', f'Flask startup: {exc}')
    stability = {}
    if ONCE:
        for archive in UPDATES.glob('*.zip'):
            stability[archive.name] = (archive.stat().st_size, archive.stat().st_mtime_ns)
    while True:
        try:
            # Watch quietly. Polling still happens on schedule but only meaningful
            # events (releases, restarts, Git changes and errors) are logged.
            # Optional SPLAR_VERBOSE_WATCH=1 restores per-poll output for debugging.
            if not ONCE and os.getenv('SPLAR_VERBOSE_WATCH', '0') == '1':
                log(f'Refresh • v{current_version()} • checking updates', 'scan')
            if not migrate_legacy_port():
                if ONCE:
                    return
                time.sleep(POLL)
                continue
            if not health():
                own_pid = recorded_pid()
                if not own_pid or not process_alive(own_pid):
                    try:
                        start_server()
                    except Exception as exc:
                        notice('flask', f'Flask unavailable: {exc}')
            watcher_iteration(stability)
        except Exception as exc:
            notice('loop', f'Watcher error: {exc}')
        if ONCE:
            return
        time.sleep(POLL)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('Watcher stopped (Flask stays running)')
    except Exception as exc:
        log(f'Fatal: {exc}')
        sys.exit(1)
PYTHON
