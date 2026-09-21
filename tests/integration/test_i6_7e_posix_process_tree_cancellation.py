"""Proves Ubuntu setsid inheritance/cancellation with a fake nested worker."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(sys.platform != 'linux' or shutil.which('setsid') is None,
                    reason='requires Linux setsid')
def test_isolated_dense_group_signal_reaches_nested_external_provider(tmp_path):
    script = tmp_path / 'supervised_fake.py'
    script.write_text('''\
import subprocess, os, sys, time
child = subprocess.Popen(['sleep','120'])
print(f'{os.getpid()} {os.getpgrp()} {child.pid}', flush=True)
time.sleep(120)
''', encoding='utf-8')
    process = subprocess.Popen([shutil.which('setsid'), sys.executable, str(script)],
                               stdout=subprocess.PIPE, text=True, start_new_session=False)
    try:
        assert process.stdout is not None
        info = process.stdout.readline().strip().split()
        assert len(info) == 3
        leader_pid, group_id, descendant_pid = map(int, info)
        assert leader_pid == group_id == process.pid
        assert group_id != os.getpgrp()
        assert os.getpgid(descendant_pid) == group_id
        os.killpg(group_id, signal.SIGTERM)
        assert process.wait(timeout=5) != 0
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            status = Path(f'/proc/{descendant_pid}/stat')
            if not status.exists() or status.read_text().split()[2] == 'Z':
                break
            time.sleep(.05)
        else:
            pytest.fail('Nested fake dense worker survived group cancellation')
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
