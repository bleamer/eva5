from os import listdir
from os.path import isfile, join

def filelist(fpath, prefix=''):
    """
        Get Filelist in the _path_ starting with _prefix_
    Args:
        fpath: File Path
        prefix: Prefix pattern for filename

    Returns: List of files

    """
    onlyfiles = [f for f in listdir(fpath) if isfile(join(fpath, f)) and f.startswith(prefix)]
    return onlyfiles
