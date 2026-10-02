"""Read/network boundary inherited by Python subprocesses during source replay."""
import os
from pathlib import Path
import sys
import sysconfig

forbidden = os.environ.get('SOURCE_RELEASE_FORBIDDEN_ROOT', '').lower().replace('\\', '/')
output = os.environ.get('SOURCE_RELEASE_OUTPUT', '').lower().replace('\\', '/')
# A repository-local virtual environment is a supported installation. Exempt
# only Python's library directories, never the surrounding environment/repo.
python_libraries = tuple({
    str(Path(sysconfig.get_path(name)).resolve()).lower().replace('\\', '/')
    for name in ('stdlib', 'platstdlib', 'purelib', 'platlib')
})


def is_original_source_read(path):
    """Keep installed dependencies separate from scientific source inputs."""
    return (forbidden and path.startswith(forbidden + '/')
            and not (output and path.startswith(output + '/'))
            and not any(path.startswith(root + '/') for root in python_libraries))


def audit(event, args):
    if event == 'open' and args and isinstance(args[0], (str, os.PathLike)):
        mode = str(args[1]) if len(args) > 1 else ''
        if 'r' in mode or mode == 'None':
            path = str(Path(args[0]).resolve()).lower().replace('\\', '/')
            if is_original_source_read(path):
                raise PermissionError('Original repository is unavailable in source-only replay: ' + path)
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise PermissionError('Network is unavailable in source-only replay')


sys.addaudithook(audit)
