"""Check installed browser PDF export before starting the full figure build.

Run explicitly with ``python tests/browser_smoke.py``. This creates only a
temporary HTML file and a disposable browser profile, never a user profile.
"""
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

import fitz


ROOT = Path(__file__).resolve().parents[1]
browser = runpy.run_path(str(ROOT/'reproduction/runtime/ca_analysis/browser_export.py'))


def main():
    results = []
    for executable in browser['find_browsers']():
        with tempfile.TemporaryDirectory(prefix='maze_browser_check_') as raw:
            folder = Path(raw).resolve()
            html, pdf = folder/'test.html', folder/'test.pdf'
            html.write_text('<!doctype html><html><body>Maze export check</body></html>', encoding='utf-8')
            command = [str(executable), *browser['platform_headless_flags'](),
                       '--headless=new', '--disable-gpu', '--no-sandbox',
                       '--no-pdf-header-footer', f'--user-data-dir={folder / "profile"}',
                       f'--print-to-pdf={pdf}', html.as_uri()]
            row = dict(browser=str(executable))
            try:
                process = subprocess.run(command, capture_output=True, timeout=30)
                row.update(exit_code=process.returncode,
                           stderr=process.stderr.decode('utf-8', errors='replace')[-2000:])
            except subprocess.TimeoutExpired as error:
                row.update(error='browser timeout',
                           stderr=(error.stderr or b'').decode('utf-8', errors='replace')[-2000:])
            row['pdf_exists'] = pdf.is_file()
            row['valid_pdf'] = False
            if pdf.is_file():
                with fitz.open(pdf) as document:
                    row['valid_pdf'] = len(document) == 1 and 'Maze export check' in document[0].get_text()
            results.append(row)
            print(json.dumps(row), flush=True)
    assert any(row.get('exit_code') == 0 and row['valid_pdf'] for row in results), results


if __name__ == '__main__':
    main()
