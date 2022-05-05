from pathlib import Path
import shutil
import pandas as pd
import pickle as pkl
import functools
import inspect
import hashlib


def hashfn(hashstr):
    return hashlib.sha224(hashstr.encode('UTF-8')).hexdigest()


def run_exists(param_dict, output_dir):
    """
    Check to see if a run matching param_dict exists.

    Parameters
    ----------
    param_dict : dict
        A dictionary that contains the key value pairs corresponding to a
        row in a table. This is usually the parameter values that specify
        a run.
    output_dir : str
        The output directory for the runs. The run table will be stored
        in this directory, as well as subdirectories corresponding to
        individual runs.

    Returns
    -------
    bool
    """
    output_dir = Path(output_dir)
    hashnum = get_run_hash(param_dict, output_dir)
    if (output_dir/str(hashnum)).exists():
        return True

def get_run_hash(param_dict, output_dir):
    """
    Get a run ID and directory that corresponds to param_dict. If a
    corresponding row of the run table does not exist, create
    a new row and generate a new directory for the run and return the
    corresponding new ID.

    Parameters
    ----------
    param_dict : dict
        A dictionary that contains the key value pairs corresponding to a
        row in a table. This is usually the parameter values that specify
        a run.
    output_dir : str
        The output directory for the runs. The run table will be stored
        in this directory, as well as subdirectories corresponding to
        individual runs.

    Returns
    -------
    int
        The hash identifying the run."""
    output_dir = Path(output_dir)
    df = pd.DataFrame(param_dict, index=[0])
    df = df.reindex(sorted(df.columns), axis=1)
    hashnum = pd.util.hash_pandas_object(df, index=False)[0]
    return hashnum
