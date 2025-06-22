"""
Utilities for filesystem operations used in the Decentralized FL Diffusion project.

This module provides helper functions to manage experiment directories and
format execution time for logging and output purposes.

Functions
---------
- reset(path): Recursively deletes all files and folders in a given directory and then removes the directory itself.
- format_time(seconds): Formats a duration in seconds into a human-readable string.
"""

import os
import shutil


def reset(path: str):
    """
    Recursively deletes all contents of the given directory and removes the directory itself.

    Parameters
    ----------
    path : str
        Path to the directory to be reset.

    Notes
    -----
    This is typically used when `overwrite_output_dir=True` is set in the configuration.
    It removes all files, subdirectories, and symbolic links before removing the root directory.
    """
    for filename in os.scandir(path):
        try:
            if os.path.isfile(filename) or os.path.islink(filename):
                os.unlink(filename)
            elif os.path.isdir(filename):
                print(f"Deleting directory {filename}")
                shutil.rmtree(filename)
        except OSError as e:
            print(f"Failed to delete {filename}. Reason: {e}")
    os.rmdir(path)


def format_time(seconds: float) -> str:
    """
    Converts a time duration given in seconds into a human-readable format.

    Parameters
    ----------
    seconds : float
        Time duration in seconds.

    Returns
    -------
    str
        A formatted string in the form of "Xh Ym Zs", "Ym Zs", or "Zs", depending on the duration.

    Examples
    --------
    >>> format_time(3661)
    '1h 1m 1s'
    >>> format_time(75)
    '1m 15s'
    >>> format_time(42)
    '42s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    if minutes > 0:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"
