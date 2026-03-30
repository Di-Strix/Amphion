from pathlib import Path
from typing import Union


def resolve_path(basepath: Union[Path, str], other: Union[Path, str] = "") -> Path:
    _basepath = Path(basepath).expanduser()
    _other = Path(other).expanduser()
    return (_basepath / _other.expanduser()).absolute()
