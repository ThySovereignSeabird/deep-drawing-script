import os
from pathlib import Path


def find_project_root(anchor_name="dataset") -> Path:
    """Climbs up directories until it finds the folder containing the anchor."""
    current_path = Path.cwd().resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / anchor_name).exists():
            return parent
    # Fallback to current working directory if not found
    return current_path

PROJECT_ROOT = find_project_root()


def get_dataset_file(fname: str):
    """Dataset directory."""
    return os.path.join(PROJECT_ROOT / "dataset", fname)
