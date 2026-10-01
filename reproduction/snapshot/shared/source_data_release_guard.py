"""Read/network boundary inherited by Python subprocesses during source replay."""
import os
from pathlib import Path
import sys

forbidden = os.environ.get('SOURCE_RELEASE_FORBIDDEN_ROOT', '').lower().replace('\\', '/')
output = os.environ.get('SOURCE_RELEASE_OUTPUT', '').lower().replace('\\', '/')


def audit(event, args):
    if event == 'open' and args and isinstance(args[0], (str, os.PathLike)):
        mode = str(args[1]) if len(args) > 1 else ''
        if 'r' in mode or mode == 'None':
            path = str(Path(args[0]).resolve()).lower().replace('\\', '/')
            if forbidden and path.startswith(forbidden + '/') and not path.startswith(output + '/'):
                raise PermissionError('Original repository is unavailable in source-only replay: ' + path)
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise PermissionError('Network is unavailable in source-only replay')


sys.addaudithook(audit)
