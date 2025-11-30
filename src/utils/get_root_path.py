from pathlib import Path

def get_root_path():
    """
    Gets the root path of the project.
    """
    return Path(__file__).resolve().parent.parent.parent