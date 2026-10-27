"""Проверяем единственный worker, очистку и повторный запуск после SIGKILL."""

import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import psutil
from worker import worker

WORKER = """
import json, os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
Path(sys.argv[1]).write_text(json.dumps([os.getpid(), child.pid]))
time.sleep(120)
"""
LAUNCH = """
import json, os, signal, sys, time
from pathlib import Path
from worker import worker
worker.ROOT = Path(sys.argv[1])
source, marker = sys.argv[3:5]
stage = os.environ['TEST_STAGE']
original_replace = Path.replace
def pause():
    Path(marker).write_text(json.dumps([os.getpid()]))
    time.sleep(120)
def replace(path, target):
    if stage == 'before_registry': pause()
    result = original_replace(path, target)
    if stage == 'after_registry': pause()
    return result
if stage in ('before_registry', 'after_registry'):
    Path.replace = replace
for event in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(event, lambda signum, _: sys.exit(128 + signum))
sys.argv = [sys.argv[0], marker]
with worker.manage_worker_processes() as logs:
    exec(compile(source, '<test-worker>', 'exec'))
"""

# Настоящие scheduler, CSV, flock и процессы; модель и Google заменены локальными заглушками.
PIPELINE = """
import json, os, subprocess, sys, threading, time
from pathlib import Path
from worker import worker
root, marker = Path(sys.argv[1]).parent, Path(sys.argv[1])
worker.ROOT = root
worker.load_config = lambda *_: None
os.environ.update(WORKER_GPU_NUM='0', WORKER_PARALLEL='2', WORKER_BACKEND='fake',
                  WORKER_NAME='recovery', WORKER_MODEL='test', WORKER_MODEL_ALIAS='',
                  WORKER_STALE_AFTER='300', WORKER_HEARTBEAT_INTERVAL='0.1')
stage = os.environ.get('TEST_STAGE', '')
original_popen = subprocess.Popen
children, barrier = [], threading.Barrier(2)
backend = None
def pause():
    marker.write_text(json.dumps([os.getpid(), backend.pid, *children]))
    time.sleep(120)
def popen(command, **kwargs):
    global backend
    if 'backends.fake' in command:
        server = 'from http.server import HTTPServer, BaseHTTPRequestHandler; import sys; HTTPServer(("127.0.0.1", int(sys.argv[1])), BaseHTTPRequestHandler).serve_forever()'
        # Для readiness достаточно любого успешного HTTP-ответа; запрос проверяется отдельно.
        server = server.replace('BaseHTTPRequestHandler).serve_forever()', 'type("Handler", (BaseHTTPRequestHandler,), {"do_GET": lambda s: (s.send_response(200), s.end_headers())})).serve_forever()')
        backend = original_popen([sys.executable, '-c', server, command[-1]], **kwargs)
        if stage == 'backend': pause()
        return backend
    return original_popen(command, **kwargs)
worker.subprocess.Popen = popen
queue = []
for index in range(2):
    row = dict.fromkeys(worker.FIELDS, '')
    row.update(job_id=f'job{index}', model='test', benchmark='gaia')
    queue.append(row)
def request(action, **values):
    if action == 'mark_stale_jobs_interrupted': return 0
    if action == 'claim': return queue.pop(0) if queue else None
    if stage == 'csv' and values['values']['status'] == 'done':
        # publish_result уже записала и fsync-нула CSV, но Google ещё не получил финал.
        pause()
worker.plan_request = request
def job(row, worker_id, logs, slot, port, interval, progress):
    script = 'import socket, sys, time; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(); time.sleep(120)'
    process = original_popen([sys.executable, '-c', script, str(port + slot)])
    children.append(process.pid)
    barrier.wait(timeout=10)
    if stage == 'parallel':
        if slot == 0: pause()
        time.sleep(120)
    if not stage and slot == 0:
        marker.write_text(json.dumps([os.getpid(), backend.pid, *children]))
    return {'job': {'job_id': row['job_id'], 'status': 'done', 'error': None, 'full_time_seconds': 1, 'time_limit_seconds': 10},
            'benchmark': {'result': 'solved', 'error': None}, 'calls': [],
            'scaffold': {'model': {'calls': 1, 'max_input_tokens': 10, 'output_tokens_sum': 2,
                                    'time_sum_seconds': 1, 'usage_complete': True}}}
worker.run_job = job
raise SystemExit(worker.execute_plan(logs))
"""


class WorkerLifecycleTest(unittest.TestCase):
    @patch("worker.worker.logging.basicConfig")
    @patch("worker.worker.archive_logs")
    def test_entrypoint_runs_in_current_process(self, archive, configure_logging):
        with patch.object(worker, 'load_config'), patch.object(worker.signal, 'signal'), \
             patch.object(worker, 'manage_worker_processes', return_value=nullcontext(Path('/logs/test'))) as lifecycle, \
             patch.object(worker, 'execute_plan', return_value=3) as run, \
             patch.object(worker.subprocess, 'Popen') as spawn, \
             patch.object(sys, 'argv', ['worker']):
            self.assertEqual(worker.main(), 3)
            lifecycle.assert_called_once_with()
            archive.assert_called_once_with()
            run.assert_called_once_with(Path('/logs/test'))
            spawn.assert_not_called()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for path in self.root.glob('_state/runtime/workers/*.json'):
            worker.stop_processes(json.loads(path.read_text()))
        self.directory.cleanup()

    def start(self, name, stage='', command=WORKER, gpu=''):
        path = self.root / f'{name}.json'
        previous = path.read_text() if path.exists() else None
        log_path = self.root / f'worker-{len(self.processes)}.log'
        with log_path.open('w') as log:
            process = subprocess.Popen(
                [sys.executable, '-c', LAUNCH, str(self.root), str(self.root / name), command, str(path)],
                env=os.environ | {'PYTHONPATH': str(worker.ROOT), 'WORKER_GPU_NUM': gpu, 'TEST_STAGE': stage,
                                  'WORKER_NAME': name, 'WORKER_BACKEND': 'fake',
                                  'WORKER_HEARTBEAT_INTERVAL': '0.05', 'WORKER_STALE_AFTER': '300'},
                stdout=log, stderr=log,
            )
        self.processes.append(process)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if path.exists() and path.read_text() != previous:
                try:
                    return process, json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
            if process.poll() is not None:
                self.fail(f'Worker exited with {process.returncode}: {log_path.read_text()}')
            time.sleep(0.05)
        self.fail('Test worker did not start')

    def assert_stopped(self, pids):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            alive = []
            for pid in pids:
                try:
                    if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                        alive.append(pid)
                except psutil.NoSuchProcess:
                    pass
            if not alive:
                return
            time.sleep(0.05)
        self.fail(f'Processes survived cleanup: {alive}')

    def expire_worker(self, pid):
        for path in self.root.glob('_state/runtime/workers/*.json'):
            record = json.loads(path.read_text())
            if record['pid'] == pid:
                record['heartbeat_at'] = time.time() - 1000
                temporary = path.with_suffix('.test-tmp')
                temporary.write_text(json.dumps(record))
                temporary.replace(path)
                return path
        self.fail(f'Worker {pid} was not registered')

    def test_new_start_leaves_workers_with_fresh_heartbeats_running(self):
        first, old = self.start('first')
        peer, peer_pids = self.start('peer')
        replacement, new = self.start('first')
        self.assertNotEqual(old, new)
        self.assertIsNone(first.poll())
        self.assertTrue(all(psutil.pid_exists(pid) for pid in old))
        self.assertIsNone(peer.poll())
        self.assertTrue(all(psutil.pid_exists(pid) for pid in peer_pids))

    def test_restart_after_sigkill_cleans_orphaned_children(self):
        first, old = self.start('first')
        first.kill()
        first.wait()
        self.assertTrue(psutil.pid_exists(old[1]))
        self.expire_worker(first.pid)
        self.start('different-name', gpu='1')
        self.assert_stopped(old)

    def test_ctrl_c_cleans_children_and_registration(self):
        process, pids = self.start('first')
        process.send_signal(signal.SIGINT)
        process.wait(timeout=12)
        self.assert_stopped(pids)
        self.assertEqual(list(self.root.glob('_state/runtime/workers/*.json')), [])

    def test_worker_has_no_supervisor_or_session_process(self):
        process, pids = self.start('first')
        self.assertEqual(process.pid, pids[0])
        self.assertEqual(psutil.Process(pids[1]).ppid(), process.pid)
        self.assertEqual({p.pid for p in psutil.Process(process.pid).children(recursive=True)}, {pids[1]})

    def test_reused_pid_is_not_killed(self):
        owner = Mock(pid=os.getpid() + 1)
        owner.create_time.return_value = 2
        with patch.object(worker.psutil, 'Process', return_value=owner), \
             patch.object(worker.psutil, 'process_iter', return_value=[]):
            worker.stop_processes({'pid': owner.pid, 'created': 1, 'token': 'old-test'})
        owner.terminate.assert_not_called()
        owner.kill.assert_not_called()

    def test_sigkill_during_registration_allows_normal_restart(self):
        for stage in ('before_registry', 'after_registry'):
            with self.subTest(stage=stage):
                process, pids = self.start(stage, stage=stage)
                process.kill()
                process.wait()
                self.assert_stopped(pids)
                if stage == 'after_registry':
                    self.expire_worker(process.pid)
                replacement, replacement_pids = self.start(stage)
                replacement.terminate()
                replacement.wait(timeout=12)
                self.assert_stopped(replacement_pids)

    def test_sigkill_pipeline_releases_processes_ports_gpu_and_preserves_csv(self):
        for stage in ('backend', 'parallel', 'csv'):
            with self.subTest(stage=stage):
                process, pids = self.start(stage, stage=stage, command=PIPELINE)
                self.assertEqual(process.pid, pids[0])
                if stage == 'parallel':
                    self.assertEqual(len(pids), 4)
                process.kill()
                process.wait()
                self.expire_worker(process.pid)
                replacement, new_pids = self.start(stage, command=PIPELINE)
                self.assertEqual(replacement.wait(timeout=15), 0)
                self.assert_stopped(pids + new_pids)
                results = sorted(self.root.glob('_state/logs/*/results.csv'), key=lambda p: p.stat().st_mtime)
                with results[-1].open() as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual({row['job_id'] for row in rows}, {'job0', 'job1'})
                self.assertTrue(all(row['status'] == 'done' for row in rows))
                if stage == 'csv':
                    with results[-2].open() as source:
                        self.assertGreaterEqual(len(list(csv.DictReader(source))), 1)

    def test_sigterm_and_terminal_hangup_clean_processes(self):
        for event in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(event=event):
                process, pids = self.start(f'signal-{event}')
                process.send_signal(event)
                process.wait(timeout=12)
                self.assert_stopped(pids)

    def test_heartbeat_updates_but_stale_cleanup_only_runs_at_startup(self):
        observer, _ = self.start('observer')
        victim, pids = self.start('victim')
        path = next(path for path in self.root.glob('_state/runtime/workers/*.json')
                    if json.loads(path.read_text())['pid'] == observer.pid)
        heartbeat = json.loads(path.read_text())['heartbeat_at']
        os.kill(victim.pid, signal.SIGSTOP)
        self.expire_worker(victim.pid)
        time.sleep(0.2)
        self.assertGreater(json.loads(path.read_text())['heartbeat_at'], heartbeat)
        self.assertIsNone(victim.poll())
        self.assertTrue(all(psutil.pid_exists(pid) for pid in pids))
        self.start('cleaner')
        victim.wait(timeout=10)
        self.assert_stopped(pids)
        self.assertIsNone(observer.poll())

    def test_concurrent_startups_clean_all_stale_workers_and_preserve_fresh(self):
        first, first_pids = self.start('stale-first')
        second, second_pids = self.start('stale-second')
        peer, _ = self.start('fresh-peer')
        for process in (first, second):
            os.kill(process.pid, signal.SIGSTOP)
            self.expire_worker(process.pid)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.start, name) for name in ('new-first', 'new-second')]
            newcomers = [future.result() for future in futures]
        for process in (first, second):
            process.wait(timeout=10)
        self.assert_stopped(first_pids + second_pids)
        self.assertIsNone(peer.poll())
        self.assertTrue(all(process.poll() is None for process, _ in newcomers))
        self.assertEqual(len(list(self.root.glob('_state/runtime/workers/*.json'))), 3)
