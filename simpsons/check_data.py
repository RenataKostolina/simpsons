import os
import subprocess
from typing import List


def dvc_pull() -> bool:
    """
    Performs a dvc pull for the specified paths with error handling.

    Args:
    target_paths: List of download paths (None for the entire repository)
    remote: The name of the remote storage (from .dvc/config)
    verbose: Whether to output a detailed log

    Returns:
    bool: Has the download been completed successfully
    """
    try:
        cmd = ["poetry", "run", "dvc", "pull"]

        result = subprocess.run(cmd, check=True, text=True, capture_output=True)

        return True

    except subprocess.CalledProcessError as e:
        print(f"DVC pull failed with error: {e.stderr}")
        return False


def ensure_data_downloaded(required_paths: List[str]) -> bool:
    """
    Checks for paths and downloads the missing ones via DVC.

    Args:
    required_paths: List of required paths (folders/files)

    Returns:
    bool: Do all paths exist after executing
    """
    missing_paths = [p for p in required_paths if not os.path.exists(p)]

    if not missing_paths:
        return True

    success = dvc_pull(target_paths=missing_paths)

    if not success:
        return False

    return True
