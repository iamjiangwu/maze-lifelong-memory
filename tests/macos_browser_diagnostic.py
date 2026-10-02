"""Temporary native-run diagnostic; removed before release."""
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

runtime = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'reproduction/runtime/ca_analysis/browser_export.py'))
browser = runtime['find_browser']()
for extra in ([], ['--disable-background-mode', '--disable-background-networking', '--disable-component-update']):
    with tempfile.TemporaryDirectory(prefix='maze_browser_diagnosis_') as raw:
        root = Path(raw).resolve()
        html = root / 'test.html'
        html.write_text('<html><body>Browser diagnostic</body></html>')
        command = [str(browser), *runtime['platform_headless_flags'](), *extra,
                   '--headless=new', '--disable-gpu', '--no-sandbox',
                   f'--user-data-dir={root / "profile"}', f'--print-to-pdf={root / "test.pdf"}', html.as_uri()]
        with (root / 'browser.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=12)
                print(json.dumps(dict(extra=extra, exit_code=process.returncode)), flush=True)
            except subprocess.TimeoutExpired:
                print(json.dumps(dict(extra=extra, timeout=True, pid=process.pid)), flush=True)
                sample = subprocess.run(['sample', str(process.pid), '1', '1'], capture_output=True, text=True, timeout=10)
                print(sample.stdout[:16000], flush=True)
                print(subprocess.run(['ps', '-o', 'pid,ppid,stat,command', '-p', str(process.pid)], capture_output=True, text=True).stdout, flush=True)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        print((root / 'browser.log').read_text(errors='replace')[-8000:], flush=True)
